import os
import asyncio
import discord
import numpy as np
import logging
from discord import app_commands
from discord.ext import voice_recv
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. Robust Monkey Patch for Opus Errors ---
# This prevents the bot from crashing if Discord sends bad audio packets.
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
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"

# Silence technical logs to keep console clean
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.ERROR)
logging.getLogger("discord.gateway").setLevel(logging.WARNING)
logging.getLogger("discord.client").setLevel(logging.WARNING)


class AudioResampler:
    @staticmethod
    def discord_to_gemini(pcm_bytes):
        # Convert Discord's 48kHz Stereo -> Gemini's 16kHz Mono
        try:
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_audio = audio_data.mean(axis=1).astype(np.int16)
            return mono_audio[::3].tobytes()
        except Exception:
            return b''

    @staticmethod
    def gemini_to_discord(pcm_bytes):
        # Convert Gemini's 24kHz Mono -> Discord's 48kHz Stereo
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

    # CRITICAL: This tells the bot "I am speaking right now"
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
    while True:
        print("Connecting to Gemini Live API...")

        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Aoede"  # Closest match to "Lyra"
                    }
                }
            },
            "system_instruction": {
                "parts": [{
                    "text": "You are Skippy. You are helpful, funny, and concise. Speak at a relaxed, slow pace."
                }]
            }
        }

        # Clear old audio so the bot doesn't hear "ghosts"
        while not receive_queue.empty():
            try:
                receive_queue.get_nowait()
            except:
                pass

        try:
            async with client_genai.aio.live.connect(model=MODEL_ID, config=config) as session:
                print("Gemini Connected! Start talking.")

                async def sender():
                    while True:
                        try:
                            audio_chunk = await asyncio.wait_for(receive_queue.get(), timeout=1.0)
                            if audio_chunk is None: continue

                            # --- ECHO GATE ---
                            # If the bot is talking, STOP LISTENING.
                            # This prevents the bot from hearing itself and cutting off.
                            if play_source.is_playing():
                                continue
                                # -----------------

                            await session.send_realtime_input(
                                audio={
                                    "data": audio_chunk,
                                    "mime_type": "audio/pcm;rate=16000"
                                }
                            )
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"Sender Error: {e}")
                            break

                async def receiver():
                    try:
                        async for response in session.receive():
                            if response.server_content is None: continue
                            model_turn = response.server_content.model_turn
                            if model_turn:
                                for part in model_turn.parts:
                                    if part.inline_data:
                                        # print(f" [GEMINI-OUT] Received {len(part.inline_data.data)} bytes")
                                        await play_source.add_audio(part.inline_data.data)
                    except Exception as e:
                        print(f"Receiver Error: {e}")

                send_task = asyncio.create_task(sender())
                recv_task = asyncio.create_task(receiver())

                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending: task.cancel()

        except asyncio.CancelledError:
            print("Session cancelled.")
            break
        except Exception as e:
            print(f"Connection Error: {e}")

        print("Session Ended. Reconnecting...")
        await asyncio.sleep(1)


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