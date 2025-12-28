import os
import asyncio
import discord
import numpy as np
import logging
from discord import app_commands
from discord.ext import voice_recv
from google import genai
from dotenv import load_dotenv

# --- 1. Robust Monkey Patch for Opus Errors ---
import discord.opus

_original_decode = discord.opus.Decoder.decode


def _patched_decode(self, *args, **kwargs):
    try:
        return _original_decode(self, *args, **kwargs)
    except discord.opus.OpusError:
        return b'\x00' * 3840


discord.opus.Decoder.decode = _patched_decode

# --- 2. Configuration & Logging ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Use the model from your snippet if available, otherwise fall back to stable exp
# MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"
MODEL_ID = "gemini-2.0-flash-exp"

logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.ERROR)
logging.getLogger("discord.gateway").setLevel(logging.WARNING)


class AudioResampler:
    @staticmethod
    def discord_to_gemini(pcm_bytes):
        # 48kHz Stereo -> 16kHz Mono
        try:
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_audio = audio_data.mean(axis=1).astype(np.int16)
            return mono_audio[::3].tobytes()
        except Exception:
            return b''

    @staticmethod
    def gemini_to_discord(pcm_bytes):
        # 24kHz Mono -> 48kHz Stereo
        try:
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            upsampled = np.repeat(audio_data, 2)
            stereo = np.column_stack((upsampled, upsampled)).flatten()
            return stereo.astype(np.int16).tobytes()
        except Exception:
            return b''


class LiveAudioSource(discord.AudioSource):
    def __init__(self):
        self.queue = asyncio.Queue()
        self._buffer = bytearray()

    async def add_audio(self, pcm_bytes):
        discord_audio = AudioResampler.gemini_to_discord(pcm_bytes)
        if discord_audio:
            await self.queue.put(discord_audio)

    def is_playing(self):
        return len(self._buffer) > 0 or not self.queue.empty()

    def read(self):
        target_size = 3840
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
        return b'\x00' * target_size

    def cleanup(self):
        pass


class DiscordToGeminiSink(voice_recv.AudioSink):
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
                # We put the DICT directly into the queue, matching your snippet's logic
                self.loop.call_soon_threadsafe(
                    self.gemini_queue.put_nowait,
                    {"data": converted, "mime_type": "audio/pcm;rate=16000"}
                )

    def cleanup(self):
        pass


# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
client_genai = genai.Client(api_key=GEMINI_API_KEY)


# --- Cleaned Up Gemini Session (Using TaskGroup) ---
async def run_gemini_session(voice_client, receive_queue, play_source):
    while True:  # Reconnect Loop
        print(f"Connecting to Gemini ({MODEL_ID})...")

        # Clear queues
        while not receive_queue.empty():
            try:
                receive_queue.get_nowait()
            except:
                pass

        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
            },
            "system_instruction": {
                "parts": [{"text": "You are Skippy. You are helpful, funny, and concise."}]
            }
        }

        try:
            async with client_genai.aio.live.connect(model=MODEL_ID, config=config) as session:
                print("Gemini Connected! Start talking.")

                async def send_realtime():
                    while True:
                        try:
                            # Wait for audio from Discord
                            msg = await asyncio.wait_for(receive_queue.get(), timeout=1.0)

                            # Echo cancellation: Don't send if bot is speaking
                            if play_source.is_playing():
                                continue

                            await session.send_realtime_input(audio=msg)
                        except asyncio.TimeoutError:
                            continue

                async def receive_audio():
                    while True:
                        async for response in session.receive():
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data:
                                        await play_source.add_audio(part.inline_data.data)

                        # If the loop finishes, the turn is over or connection closed.
                        print("Receive loop ended (Turn complete or connection drop)")
                        break

                # Use Python 3.11+ TaskGroup for better stability
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(send_realtime())
                    tg.create_task(receive_audio())

        except Exception as e:
            print(f"Session Error: {e}")

        print("Session Ended. Reconnecting in 1 second...")
        await asyncio.sleep(1)


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