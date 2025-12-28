import os
import asyncio
import discord
import numpy as np
import scipy.signal
from discord import app_commands
from discord.ext import voice_recv
from google import genai
from dotenv import load_dotenv
import time

# --- Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# The specialized model for Live API
MODEL_ID = "gemini-2.0-flash-exp"

# --- Audio Constants ---
# Discord sends/receives 48kHz Stereo (2 channels)
DISCORD_RATE = 48000
DISCORD_CHANNELS = 2

# Gemini Live API expects 16kHz Mono (1 channel) for input
# and sends 24kHz Mono (1 channel) for output
GEMINI_INPUT_RATE = 16000
GEMINI_OUTPUT_RATE = 24000
FRAME_SIZE = 960  # Discord frame size (20ms)


class AudioResampler:
    """Handles Real-time PCM resampling between Discord and Gemini."""

    @staticmethod
    def discord_to_gemini(pcm_bytes):
        """
        Convert Discord (48kHz Stereo 16-bit) -> Gemini (16kHz Mono 16-bit).
        Process: Merge Stereo to Mono -> Downsample 48k to 16k.
        """
        # Convert bytes to Int16 Array
        audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Reshape to (Samples, Channels)
        audio_data = audio_data.reshape(-1, 2)

        # 1. Stereo to Mono (Average the channels)
        mono_audio = audio_data.mean(axis=1).astype(np.int16)

        # 2. Resample 48k -> 16k (Decimate by 3)
        # Simple slicing [::3] is fast and 'good enough' for voice
        # For production quality, use scipy.signal.resample
        resampled_audio = mono_audio[::3]

        return resampled_audio.tobytes()

    @staticmethod
    def gemini_to_discord(pcm_bytes):
        """
        Convert Gemini (24kHz Mono 16-bit) -> Discord (48kHz Stereo 16-bit).
        Process: Upsample 24k to 48k -> Duplicate to Stereo.
        """
        audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)

        # 1. Resample 24k -> 48k (Upsample by 2)
        # Linear interpolation is better than simple repetition for upsampling
        # But for speed in python, we can just repeat elements
        upsampled = np.repeat(audio_data, 2)

        # 2. Mono to Stereo (Duplicate the array into 2 columns)
        stereo = np.column_stack((upsampled, upsampled)).flatten()

        return stereo.astype(np.int16).tobytes()


class LiveAudioSource(discord.AudioSource):
    """
    A never-ending AudioSource that plays data from a queue.
    If queue is empty, returns silence (to keep connection open).
    """

    def __init__(self):
        self.queue = asyncio.Queue()
        self.talking = False
        self._buffer = bytearray()

    async def add_audio(self, pcm_bytes):
        """Accepts raw Gemini audio (24kHz Mono)"""
        # Convert to Discord format immediately upon receipt
        discord_audio = AudioResampler.gemini_to_discord(pcm_bytes)
        await self.queue.put(discord_audio)

    def read(self):
        # Return 20ms of audio (3840 bytes for 48k stereo 16bit)
        target_size = 3840

        # Fill buffer from queue
        while len(self._buffer) < target_size:
            try:
                # We use get_nowait() because read() is not async
                chunk = self.queue.get_nowait()
                self._buffer.extend(chunk)
            except asyncio.QueueEmpty:
                break

        if len(self._buffer) >= target_size:
            data = self._buffer[:target_size]
            self._buffer = self._buffer[target_size:]
            return bytes(data)
        else:
            # Return silence if no data available
            return b'\x00' * target_size

    def cleanup(self):
        pass


class DiscordToGeminiSink(voice_recv.AudioSink):
    """Captures Discord audio and pushes it to the Gemini Loop."""

    def __init__(self, loop, gemini_queue):
        self.loop = loop
        self.gemini_queue = gemini_queue

    def wants_opus(self):
        return False  # We want PCM

    def write(self, user, data):
        # 'data' is a UserData object containing .pcm
        # We need to run this thread-safe
        if user is None: return  # Ignore unknown sources

        # Simple Voice Activity Detection (VAD) by volume
        # (Optional: prevents sending total silence)
        pcm = data.pcm
        if len(pcm) > 0:
            converted = AudioResampler.discord_to_gemini(pcm)
            self.loop.call_soon_threadsafe(self.gemini_queue.put_nowait, converted)

    def cleanup(self):
        pass


# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Global Client for GenAI
client_genai = genai.Client(api_key=GEMINI_API_KEY)


# --- The Gemini Live Session Logic ---
async def run_gemini_session(voice_client, receive_queue, play_source):
    """
    Manages the persistent WebSocket connection to Gemini.
    """
    print("Connecting to Gemini Live API...")

    config = {
        "response_modalities": ["AUDIO"],
    }

    try:
        async with client_genai.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("Gemini Connected! Start talking.")

            # Task 1: Sender (Discord -> Gemini)
            async def sender():
                while True:
                    audio_chunk = await receive_queue.get()
                    # Streaming input to Gemini
                    await session.send(input={"data": audio_chunk, "mime_type": "audio/pcm;rate=16000"},
                                       end_of_turn=False)

            # Task 2: Receiver (Gemini -> Discord)
            async def receiver():
                async for response in session.receive():
                    # Check if response has audio data
                    if response.server_content is None:
                        continue

                    model_turn = response.server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.inline_data:
                                # We got audio bytes!
                                await play_source.add_audio(part.inline_data.data)

            # Run both concurrently
            send_task = asyncio.create_task(sender())
            recv_task = asyncio.create_task(receiver())

            # Wait until one fails or we cancel
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()

    except Exception as e:
        print(f"Session Error: {e}")
    finally:
        print("Gemini Session Ended.")


# --- Commands ---

@tree.command(name="live", description="Start a live, interruptible conversation!")
async def live(interaction: discord.Interaction):
    if not interaction.user.voice:
        await interaction.response.send_message("Join a voice channel first!", ephemeral=True)
        return

    await interaction.response.defer()

    # 1. Connect to Voice
    vc = interaction.guild.voice_client
    if not vc:
        vc = await interaction.user.voice.channel.connect(cls=voice_recv.VoiceRecvClient)

    # 2. Setup Queues & Sources
    gemini_input_queue = asyncio.Queue()  # Discord -> Gemini
    output_source = LiveAudioSource()  # Gemini -> Discord

    # 3. Start Listening (Capture User Audio)
    # We pass the event loop and queue to the Sink
    vc.listen(DiscordToGeminiSink(client.loop, gemini_input_queue))

    # 4. Start Playing (Output Source)
    # We play the custom source immediately. It will play silence until Gemini speaks.
    vc.play(output_source)

    # 5. Launch Gemini Session Background Task
    # We store the task on the vc object to cancel it later if needed
    vc.gemini_task = client.loop.create_task(
        run_gemini_session(vc, gemini_input_queue, output_source)
    )

    await interaction.followup.send("Skippy is listening! (Disconnect to stop)")


@tree.command(name="stop", description="Stop the live session.")
async def stop(interaction: discord.Interaction):
    if interaction.guild.voice_client:
        if hasattr(interaction.guild.voice_client, 'gemini_task'):
            interaction.guild.voice_client.gemini_task.cancel()

        await interaction.guild.voice_client.disconnect()
        await interaction.response.send_message("Session ended.")
    else:
        await interaction.response.send_message("Not connected.", ephemeral=True)


@client.event
async def on_ready():
    await tree.sync()
    print(f"Skippy Live (Real-Time) is Ready.")


client.run(DISCORD_TOKEN)