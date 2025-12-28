import os
import asyncio
import discord
import numpy as np
import logging
from discord import app_commands
from discord.ext import voice_recv
from google import genai
from google.genai import types  # Added for safer typing
from dotenv import load_dotenv

# --- 1. Robust Monkey Patch for Opus Errors ---
# This patches the standard discord.py decoder to ignore corruption errors.
# It works by wrapping the low-level decode method.
import discord.opus

_original_decode = discord.opus.Decoder.decode


def _patched_decode(self, *args, **kwargs):
    try:
        return _original_decode(self, *args, **kwargs)
    except discord.opus.OpusError:
        # When a corrupted packet is received, return 20ms of silence (stereo, 48k)
        # to keep the audio stream flowing without crashing the bot.
        # 960 samples * 2 channels * 2 bytes = 3840 bytes
        return b'\x00' * 3840


discord.opus.Decoder.decode = _patched_decode

# --- 2. Configuration & Logging ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_ID = "gemini-2.0-flash-exp"

# Silence the standard RTCP noise from the voice receiver
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.ERROR)

# Audio Constants
DISCORD_RATE = 48000
GEMINI_INPUT_RATE = 16000
GEMINI_OUTPUT_RATE = 24000


class AudioResampler:
    """Handles Real-time PCM resampling between Discord and Gemini."""

    @staticmethod
    def discord_to_gemini(pcm_bytes):
        # Discord (48k Stereo) -> Gemini (16k Mono)
        try:
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_audio = audio_data.mean(axis=1).astype(np.int16)  # Stereo -> Mono
            return mono_audio[::3].tobytes()  # 48k -> 16k (Decimate)
        except Exception:
            return b''

    @staticmethod
    def gemini_to_discord(pcm_bytes):
        # Gemini (24k Mono) -> Discord (48k Stereo)
        try:
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            upsampled = np.repeat(audio_data, 2)  # 24k -> 48k
            stereo = np.column_stack((upsampled, upsampled)).flatten()  # Mono -> Stereo
            return stereo.astype(np.int16).tobytes()
        except Exception:
            return b''


class LiveAudioSource(discord.AudioSource):
    """Never-ending source that plays from a queue or silence."""

    def __init__(self):
        self.queue = asyncio.Queue()
        self._buffer = bytearray()

    async def add_audio(self, pcm_bytes):
        discord_audio = AudioResampler.gemini_to_discord(pcm_bytes)
        if discord_audio:
            await self.queue.put(discord_audio)

    def read(self):
        target_size = 3840  # 20ms of 48k stereo 16bit
        while len(self._buffer) < target_size:
            try:
                chunk = self.queue.get_nowait()
                self._buffer.extend(chunk)
            except asyncio.QueueEmpty:
                break

        if len(self._buffer) >= target_size:
            data = self._buffer[:target_size]
            self._buffer = self._buffer[target_size:]
            return bytes(data)
        return b'\x00' * target_size  # Silence

    def cleanup(self):
        pass


class DiscordToGeminiSink(voice_recv.AudioSink):
    """Captures Discord audio to Gemini."""

    def __init__(self, loop, gemini_queue):
        self.loop = loop
        self.gemini_queue = gemini_queue

    def wants_opus(self):
        return False

    def write(self, user, data):
        if user is None: return
        pcm = data.pcm
        if len(pcm) > 0:
            converted = AudioResampler.discord_to_gemini(pcm)
            if converted:
                self.loop.call_soon_threadsafe(self.gemini_queue.put_nowait, converted)

    def cleanup(self):
        pass


# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
client_genai = genai.Client(api_key=GEMINI_API_KEY)


# --- Gemini Session ---
async def run_gemini_session(voice_client, receive_queue, play_source):
    print("Connecting to Gemini Live API...")

    # Config for the session
    config = {"response_modalities": ["AUDIO"]}

    try:
        async with client_genai.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("Gemini Connected! Start talking.")

            async def sender():
                while True:
                    audio_chunk = await receive_queue.get()
                    if audio_chunk is None: continue

                    # FIXED: Use send_realtime_input with correct 'audio' kwarg
                    # This avoids the DeprecationWarning and the TypeError
                    await session.send_realtime_input(
                        audio={
                            "data": audio_chunk,
                            "mime_type": "audio/pcm"
                        }
                    )

            async def receiver():
                async for response in session.receive():
                    if response.server_content is None: continue
                    model_turn = response.server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.inline_data:
                                await play_source.add_audio(part.inline_data.data)

            send_task = asyncio.create_task(sender())
            recv_task = asyncio.create_task(receiver())

            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending: task.cancel()

    except Exception as e:
        print(f"Session Error: {e}")
    finally:
        print("Gemini Session Ended.")


# --- Commands ---
@tree.command(name="live", description="Start a live conversation!")
async def live(interaction: discord.Interaction):
    if not interaction.user.voice:
        await interaction.response.send_message("Join a voice channel first!", ephemeral=True)
        return

    await interaction.response.defer()
    vc = interaction.guild.voice_client
    if not vc:
        vc = await interaction.user.voice.channel.connect(cls=voice_recv.VoiceRecvClient)

    gemini_input_queue = asyncio.Queue()
    output_source = LiveAudioSource()

    vc.listen(DiscordToGeminiSink(client.loop, gemini_input_queue))
    vc.play(output_source)

    vc.gemini_task = client.loop.create_task(
        run_gemini_session(vc, gemini_input_queue, output_source)
    )

    await interaction.followup.send("Skippy is listening!")


@tree.command(name="stop", description="Stop session.")
async def stop(interaction: discord.Interaction):
    if interaction.guild.voice_client:
        if hasattr(interaction.guild.voice_client, 'gemini_task'):
            interaction.guild.voice_client.gemini_task.cancel()
        await interaction.guild.voice_client.disconnect()
        await interaction.response.send_message("Session ended.")


@client.event
async def on_ready():
    await tree.sync()
    print(f"Skippy Live is Ready.")


client.run(DISCORD_TOKEN)