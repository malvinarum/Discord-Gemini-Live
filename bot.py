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
# This prevents the bot from crashing if Discord sends empty audio packets.
import discord.opus

_original_decode = discord.opus.Decoder.decode


def _patched_decode(self, *args, **kwargs):
    try:
        return _original_decode(self, *args, **kwargs)
    except discord.opus.OpusError:
        return b'\x00' * 3840


discord.opus.Decoder.decode = _patched_decode

# --- 2. Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_ID = os.getenv('GEMINI_MODEL_ID', "gemini-2.5-flash-native-audio-preview-12-2025")
VOICE_NAME = os.getenv('GEMINI_VOICE_NAME', "Aoede")
# Default to a generic assistant if no personality is provided
SYSTEM_PROMPT = os.getenv('BOT_PERSONALITY', "You are a helpful AI assistant.")

# --- 3. Logging Setup ---
# Suppress noisy INFO logs from the voice receiver and gateway
logging.getLogger("discord.ext.voice_recv").setLevel(logging.ERROR)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.ERROR)
logging.getLogger("discord.gateway").setLevel(logging.WARNING)
logging.getLogger("discord.client").setLevel(logging.WARNING)


class AudioResampler:
    @staticmethod
    def discord_to_gemini(pcm_bytes):
        # Discord (48k Stereo) -> Gemini (16k Mono)
        try:
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_audio = audio_data.mean(axis=1).astype(np.int16)
            return mono_audio[::3].tobytes()
        except Exception:
            return b''

    @staticmethod
    def gemini_to_discord(pcm_bytes):
        # Gemini (24k Mono) -> Discord (48k Stereo)
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
        target_size = 3840  # 20ms of audio
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
                self.loop.call_soon_threadsafe(self.gemini_queue.put_nowait, converted)

    def cleanup(self):
        pass


# --- Bot Initialization ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
client_genai = genai.Client(api_key=GEMINI_API_KEY)


async def set_presence(state):
    """
    Updates the bot's status to indicate state without triggering Discord ratelimits.
    'active' = Green (Listening)
    'idle' = Yellow (Waiting)
    """
    if state == "active":
        activity = discord.Activity(type=discord.ActivityType.listening, name="to voice")
        status = discord.Status.online
    else:
        activity = discord.Activity(type=discord.ActivityType.custom, name="status", state="Waiting for call...")
        status = discord.Status.idle

    await client.change_presence(status=status, activity=activity)


async def run_gemini_session(voice_client, receive_queue, play_source):
    while True:
        await set_presence("active")
        print(f"[Session] Connecting to model: {MODEL_ID}")

        # Flush queue to prevent old audio processing
        while not receive_queue.empty():
            try:
                receive_queue.get_nowait()
            except:
                pass

        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": VOICE_NAME}}
            },
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]}
        }

        try:
            async with client_genai.aio.live.connect(model=MODEL_ID, config=config) as session:
                print("[Session] Connected. Listening...")

                async def sender():
                    buffer = bytearray()
                    # Accumulate ~150ms of audio to prevent network jitter
                    THRESHOLD = 4800
                    # Send 10ms of silence to keep the websocket alive when the user isn't talking
                    SILENCE_CHUNK = b'\x00' * 320

                    while True:
                        try:
                            try:
                                # Wait for audio from Discord (short timeout)
                                msg = await asyncio.wait_for(receive_queue.get(), timeout=0.1)

                                # Echo Cancellation / Barge-In Logic
                                if play_source.is_playing():
                                    buffer.clear()
                                    # Send silence so Google knows we are still here
                                    await session.send_realtime_input(
                                        audio={"data": SILENCE_CHUNK, "mime_type": "audio/pcm"})
                                    continue

                                buffer.extend(msg)
                                if len(buffer) >= THRESHOLD:
                                    await session.send_realtime_input(
                                        audio={"data": bytes(buffer), "mime_type": "audio/pcm"})
                                    buffer.clear()

                            except asyncio.TimeoutError:
                                # If buffer has data, flush it
                                if len(buffer) > 0:
                                    await session.send_realtime_input(
                                        audio={"data": bytes(buffer), "mime_type": "audio/pcm"})
                                    buffer.clear()
                                else:
                                    # Otherwise send Keep-Alive silence
                                    await session.send_realtime_input(
                                        audio={"data": SILENCE_CHUNK, "mime_type": "audio/pcm"})
                                continue
                        except Exception as e:
                            print(f"[Sender] Error: {e}")
                            break

                async def receiver():
                    while True:
                        try:
                            async for response in session.receive():
                                if response.server_content and response.server_content.model_turn:
                                    for part in response.server_content.model_turn.parts:
                                        if part.inline_data:
                                            await play_source.add_audio(part.inline_data.data)
                        except Exception as e:
                            print(f"[Receiver] Error: {e}")
                            break
                        break

                send_task = asyncio.create_task(sender())
                recv_task = asyncio.create_task(receiver())

                # Wait for either task to fail/finish
                done, pending = await asyncio.wait([send_task, recv_task], return_when=asyncio.FIRST_COMPLETED)
                for task in pending: task.cancel()

        except Exception as e:
            print(f"[Session] Connection Error: {e}")

        print("[Session] Reconnecting in 2 seconds...")
        await asyncio.sleep(2)


@tree.command(name="live", description="Start a live voice conversation.")
async def live(interaction: discord.Interaction):
    if not interaction.user.voice:
        await interaction.response.send_message("Please join a voice channel first.", ephemeral=True)
        return
    await interaction.response.defer()

    vc = interaction.guild.voice_client
    if not vc:
        vc = await interaction.user.voice.channel.connect(cls=voice_recv.VoiceRecvClient)

    gemini_queue = asyncio.Queue()
    output_source = LiveAudioSource()
    vc.listen(DiscordToGeminiSink(client.loop, gemini_queue))
    vc.play(output_source)
    vc.gemini_task = client.loop.create_task(run_gemini_session(vc, gemini_queue, output_source))
    await interaction.followup.send("AI is listening! Start speaking.")


@tree.command(name="stop", description="Stop the live session.")
async def stop(interaction: discord.Interaction):
    if interaction.guild.voice_client:
        if hasattr(interaction.guild.voice_client, 'gemini_task'):
            interaction.guild.voice_client.gemini_task.cancel()
        await interaction.guild.voice_client.disconnect()
        await set_presence("idle")
        await interaction.response.send_message("Session ended.")


@client.event
async def on_ready():
    await tree.sync()
    await set_presence("idle")
    print(f"Bot is ready and logged in as {client.user}.")


client.run(DISCORD_TOKEN)