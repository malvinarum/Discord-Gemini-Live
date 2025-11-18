import os
import discord
from discord import app_commands
import google.generativeai as genai
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.cloud import speech
import wave
import time
import asyncio
import subprocess
from discord.ext import voice_recv

# --- 1. Load Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BOT_PERSONALITY = os.getenv('BOT_PERSONALITY', 'You are a helpful assistant.')
TTS_VOICE_NAME = os.getenv('TTS_VOICE_NAME', 'en-US-WaveNet-D')

GOOGLE_SERVICE_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if GOOGLE_SERVICE_JSON and os.path.exists(GOOGLE_SERVICE_JSON):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_SERVICE_JSON
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set. TTS/STT will fail.")

# --- 2. Configure Gemini ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 512,
        "response_mime_type": "text/plain",
    }
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
        system_instruction=BOT_PERSONALITY,
    )
    print("Gemini configured.")
except Exception as e:
    print(f"Error configuring Gemini: {e}")

# --- 3. Configure Cloud Clients ---
try:
    tts_client = texttospeech.TextToSpeechClient()
    stt_client = speech.SpeechClient()
    print("Google Cloud clients configured.")
except Exception as e:
    print(f"Error configuring Cloud clients: {e}")

# --- 4. Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# --- 5. Session Management ---
# Stores per-guild session data: { guild_id: { 'active': bool, 'chat': ChatSession, 'voice': VoiceClient } }
sessions = {}


@client.event
async def on_ready():
    await tree.sync()
    print(f'Skippy (v3.0 - Conversational) is ready as {client.user}')


# --- 6. Core Logic: The Conversation Loop ---

async def listen_loop(guild_id, channel):
    """
    The main heartbeat of the conversational bot.
    It records -> checks for speech -> responds -> repeats.
    """
    session = sessions.get(guild_id)
    if not session or not session['active']:
        print(f"Session inactive for guild {guild_id}. Stopping loop.")
        return

    voice_client = session['voice']

    # 1. Don't listen if we are currently speaking!
    while voice_client.is_playing():
        await asyncio.sleep(0.5)

    # 2. Setup recording
    filename = f"session_{guild_id}_{int(time.time())}.wav"
    sink = voice_recv.WaveSink(filename)

    print(f"Listening... ({filename})")

    # We need a way to pass data out of the callback, so we use a future
    loop = asyncio.get_running_loop()
    recording_finished_future = loop.create_future()

    def callback(sink, exc=None):
        if not recording_finished_future.done():
            recording_finished_future.set_result(sink.destination)

    # 3. Start listening
    voice_client.listen(sink, after=lambda e: callback(sink, e))

    # 4. Wait for X seconds (Silence detection is hard, so we use fixed chunks for now)
    #    If the user talks, we catch it. If not, we process silence (and ignore it).
    try:
        await asyncio.sleep(5.0)  # Listen for 5 seconds chunks
        voice_client.stop_listening()  # This triggers the callback

        # Wait for the file to be ready
        recorded_filename = await asyncio.wait_for(recording_finished_future, timeout=5.0)

        # 5. Process the audio
        await process_conversation_turn(guild_id, channel, recorded_filename)

    except Exception as e:
        print(f"Error in listen loop: {e}")
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass

    # 6. RECURSIVE CALL: Immediately loop back to listening
    #    We use create_task to avoid recursion depth limits (stack overflow)
    if sessions.get(guild_id, {}).get('active'):
        asyncio.create_task(listen_loop(guild_id, channel))


async def process_conversation_turn(guild_id, channel, filename):
    """
    Handles one "turn" of the conversation: STT -> Think -> TTS.
    """
    session = sessions.get(guild_id)
    if not session: return

    mono_filename = filename.replace(".wav", "_mono.wav")

    try:
        # A. Convert to Mono
        success = await client.loop.run_in_executor(None, convert_to_mono, filename, mono_filename)
        if not success: return

        # B. Transcribe
        text = await client.loop.run_in_executor(None, transcribe_audio, mono_filename)

        # --- SILENCE CHECK ---
        # If text is empty/short, user probably didn't speak. We just ignore it.
        if not text or len(text.strip()) < 2:
            print("Silence/Noise detected. Ignoring.")
            return

        print(f"User said: {text}")

        # C. Gemini (Contextual)
        # We use the PERSISTENT chat session stored in the global dict
        chat = session['chat']
        response = await chat.send_message_async(text)
        reply_text = response.text
        print(f"Skippy: {reply_text}")

        # D. Speak
        await speak_response(session['voice'], reply_text)

    except Exception as e:
        print(f"Turn error: {e}")
    finally:
        # Cleanup files
        for f in [filename, mono_filename]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass


# --- 7. Helper Functions ---

def convert_to_mono(in_file, out_file):
    try:
        subprocess.run(
            ["ffmpeg", "-i", in_file, "-ac", "1", "-y", out_file],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False


def transcribe_audio(filename):
    try:
        with open(filename, "rb") as f:
            content = f.read()
        with wave.open(filename, "rb") as w:
            rate = w.getframerate()
            ch = w.getnchannels()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=rate,
            language_code="en-US",
            audio_channel_count=ch
        )
        resp = stt_client.recognize(config=config, audio=audio)
        if resp.results: return resp.results[0].alternatives[0].transcript
    except Exception as e:
        print(f"STT Error: {e}")
    return None


async def speak_response(voice_client, text):
    if not voice_client or not voice_client.is_connected(): return

    try:
        s_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code="en-US", name=TTS_VOICE_NAME
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        resp = await client.loop.run_in_executor(
            None, tts_client.synthesize_speech,
            texttospeech.SynthesizeSpeechRequest(input=s_input, voice=voice_params, audio_config=audio_config)
        )

        fname = f"reply_{int(time.time())}.mp3"
        with open(fname, "wb") as f:
            f.write(resp.audio_content)

        # Stop listening before speaking (handled in loop, but safe to double check)
        if voice_client.is_listening(): voice_client.stop_listening()
        if voice_client.is_playing(): voice_client.stop()

        voice_client.play(
            discord.FFmpegPCMAudio(fname),
            after=lambda e: cleanup_file(fname)
        )

        # Wait for speech to finish so we don't record ourselves
        while voice_client.is_playing():
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"TTS Error: {e}")


def cleanup_file(fname):
    if os.path.exists(fname):
        try:
            os.remove(fname)
        except:
            pass


# --- 8. Commands ---

@tree.command(name="start_chat", description="Start a continuous conversation session.")
async def start_chat(interaction: discord.Interaction):
    await interaction.response.defer()

    if not interaction.user.voice:
        await interaction.followup.send("Join a voice channel first!")
        return

    guild_id = interaction.guild_id

    # Connect
    try:
        vc = await interaction.user.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
    except Exception:
        vc = interaction.guild.voice_client
        if not vc:
            await interaction.followup.send("Could not connect.")
            return

    # Initialize Session
    chat_session = gemini_model.start_chat(history=[])
    sessions[guild_id] = {
        'active': True,
        'chat': chat_session,
        'voice': vc
    }

    await interaction.followup.send("Session started! I'm listening... (Use /stop_chat to end)")

    # Kick off the loop
    asyncio.create_task(listen_loop(guild_id, interaction.channel))


@tree.command(name="stop_chat", description="End the conversation.")
async def stop_chat(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    if guild_id in sessions:
        sessions[guild_id]['active'] = False  # This kills the loop

        vc = sessions[guild_id]['voice']
        if vc: await vc.disconnect()

        del sessions[guild_id]
        await interaction.response.send_message("Session ended. Nap time.")
    else:
        await interaction.response.send_message("No active session.")


# --- Run ---
client.run(DISCORD_TOKEN)