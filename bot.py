import os
import discord
from discord import app_commands
from google import genai  # <--- NEW SDK IMPORT PATTERN
from google.genai import types  # <--- NEW TYPES IMPORT
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.cloud import speech
import wave
import time
import asyncio
import functools
from discord.ext import voice_recv

# --- 1. Load Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# Load the complex persona
BOT_PERSONALITY = os.getenv('BOT_PERSONALITY', 'You are a helpful, witty, and concise assistant.')
TTS_VOICE_NAME = os.getenv('TTS_VOICE_NAME', 'en-US-WaveNet-D')

GOOGLE_SERVICE_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if GOOGLE_SERVICE_JSON and os.path.exists(GOOGLE_SERVICE_JSON):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_SERVICE_JSON
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set or file not found. TTS/STT will fail.")

# --- 2. Configure Gemini (New SDK Pattern) ---
try:
    # Initialize the Client (No more global configure)
    client_genai = genai.Client(api_key=GEMINI_API_KEY)

    # --- VOICE SIMULATION GUIDANCE ---
    voice_guidance = """
    --- Voice Simulation Guidance ---
    Your final responses must be written for a Text-to-Speech (TTS) engine, simulating a live, human conversation.
    1. CONVERSATIONAL FLOW: Use a casual, informal, and interruptible speaking style.
    2. PACING: Keep sentences concise. Use short sentences and natural transitions.
    3. INFLECTION & PAUSING: Use proper and varied punctuation (Ellipses ..., Em dashes —, Commas ,).
    4. AVOID LISTS: Never use bullet points or numbered lists.
    5. FORMATTING: Do not use Markdown formatting like bolding or italics. Use only plain text.
    """

    system_instruction = BOT_PERSONALITY + voice_guidance

    # Define the config object once
    generation_config = types.GenerateContentConfig(
        temperature=0.8,
        top_p=0.95,
        top_k=64,
        max_output_tokens=512,
        response_mime_type="text/plain",
        system_instruction=system_instruction,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
        ]
    )

    # Set the model name
    # We will try to use this model name in the client calls
    GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"  # Fallback to a known working model if 3-preview fails
    # Note: For the new SDK, we pass the model name during the generate call, not init.

    print(f"Gemini Client initialized. Target Model: {GEMINI_MODEL_NAME} (or compatible)")

except Exception as e:
    print(f"FATAL Error configuring Gemini Client: {e}")
    exit()

# --- 3. Configure Google Cloud TTS ---
try:
    tts_client = texttospeech.TextToSpeechClient()
    print("Google Cloud TTS client configured successfully.")
except Exception as e:
    print(f"Error configuring Google Cloud TTS: {e}")

# --- 3.5. Configure Google Cloud STT ---
try:
    stt_client = speech.SpeechClient()
    print("Google Cloud STT client configured successfully.")
except Exception as e:
    print(f"Error configuring Google Cloud STT: {e}")

# --- 4. Configure Discord Bot ---
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# --- HELPER: Run Sync Gemini Calls in Thread ---
async def generate_skippy_response(prompt_text: str):
    """
    Wraps the synchronous Gemini Client call in an executor to prevent
    blocking the Discord bot loop.
    """

    def _call_gemini():
        try:
            # Try Gemini 3 Preview first
            try:
                response = client_genai.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=prompt_text,
                    config=generation_config
                )
                return response
            except Exception:
                # Fallback to 2.0 Flash or Pro if 3 is not available
                print("Gemini 3 preview failed/not found, falling back...")
                response = client_genai.models.generate_content(
                    model="gemini-2.0-flash-exp",  # Or gemini-1.5-pro
                    contents=prompt_text,
                    config=generation_config
                )
                return response
        except Exception as inner_e:
            print(f"Gemini generation failed: {inner_e}")
            return None

    return await client.loop.run_in_executor(None, _call_gemini)


# --- 5. Bot Events ---

@client.event
async def on_ready():
    try:
        await tree.sync()
        print("Command tree synced.")
    except Exception as e:
        print(f"Error syncing command tree: {e}")

    print(f'Skippy (New SDK Edition) is online. Logged in as {client.user}')
    await client.change_presence(activity=discord.Game(name="Judging your life choices..."))


# --- 6. Slash Commands (Text) ---

@tree.command(name="ask", description="Ask a question to the Gemini AI.")
async def ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()

    try:
        print(f"User '{interaction.user.name}' asked: {prompt}")

        # Use our helper function
        response = await generate_skippy_response(prompt)

        gemini_response = ""
        if response and response.text:
            gemini_response = response.text
        else:
            # Handling Safety/Block/Error
            gemini_response = "Hmph. The cosmic censors blocked me. Ask something less foolish, wot not?"
            if response and response.candidates:
                print(f"Blocked. Finish reason: {response.candidates[0].finish_reason}")

        await send_long_message(interaction, f"**Skippy:** {gemini_response}")

    except Exception as e:
        print(f"An error occurred while processing Gemini request: {e}")
        await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)


# --- 7. Slash Commands (Voice) ---

@tree.command(name="join", description="Joins your current voice channel.")
async def join(interaction: discord.Interaction):
    await interaction.response.defer()

    if not interaction.user.voice:
        await interaction.followup.send("You're not in a voice channel, meat-bag.", ephemeral=True)
        return

    voice_channel = interaction.user.voice.channel

    if interaction.guild.voice_client:
        if interaction.guild.voice_client.channel == voice_channel:
            await interaction.followup.send("I'm already *in* your channel.", ephemeral=True)
            return
        try:
            await interaction.guild.voice_client.move_to(voice_channel)
            await interaction.followup.send(f"Fine, moving to `{voice_channel.name}`.")
        except Exception as e:
            await interaction.followup.send(f"Problem moving: {e}")
        return

    try:
        await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        await interaction.followup.send(f"Okay, I'm in `{voice_channel.name}`.")
    except Exception as e:
        await interaction.followup.send(f"Error connecting: {e}", ephemeral=True)


@tree.command(name="leave", description="Leaves the voice channel.")
async def leave(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    if not interaction.guild.voice_client:
        await interaction.followup.send("I'm not in a voice channel.")
        return
    try:
        await interaction.guild.voice_client.disconnect()
        await interaction.followup.send("Leaving. Good riddance.")
    except Exception as e:
        await interaction.followup.send(f"Problem leaving: {e}")


# --- 8. Slash Command (TTS Test) ---

@tree.command(name="say", description="Speaks the given text.")
async def say(interaction: discord.Interaction, text: str):
    if not interaction.user.voice:
        await interaction.response.send_message("Join a voice channel first.", ephemeral=True)
        return

    await interaction.response.defer()

    # Ensure connection logic is robust (copy from chat/join if needed)
    voice_client = interaction.guild.voice_client
    if not voice_client:
        try:
            voice_client = await interaction.user.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        except Exception as e:
            await interaction.followup.send(f"Failed to connect: {e}")
            return

    try:
        await speak_text(interaction, text)
        await interaction.followup.send(f"I said: \"{text}\"")
    except Exception as e:
        await interaction.followup.send(f"Error speaking: {e}")


# --- 9. Slash Command (Core Loop) ---

async def stop_listening_after(voice_client, delay: float):
    await asyncio.sleep(delay)
    print(f"Timeout reached. Stopping listening.")
    voice_client.stop_listening()


@tree.command(name="chat", description="Have a one-shot voice conversation.")
async def chat(interaction: discord.Interaction):
    if not interaction.user.voice:
        await interaction.response.send_message("Join a voice channel first.", ephemeral=True)
        return

    voice_channel = interaction.user.voice.channel
    await interaction.response.defer()

    voice_client = interaction.guild.voice_client
    if not voice_client:
        try:
            voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        except Exception as e:
            await interaction.followup.send(f"Failed to connect: {e}")
            return
    elif not isinstance(voice_client, voice_recv.VoiceRecvClient):
        await voice_client.disconnect()
        voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)

    if voice_client.is_playing():
        voice_client.stop()
    voice_client.stop_listening()

    try:
        await speak_text(interaction, "I'm listening...")
        await interaction.followup.send("I'm listening... (10s max)")
        while voice_client.is_playing():
            await asyncio.sleep(0.1)
    except Exception as e:
        await interaction.followup.send(f"Error speaking: {e}")
        return

    filename = f"rec_{interaction.id}_{int(time.time())}.wav"
    sink = voice_recv.WaveSink(filename)

    voice_client.listen(sink, after=lambda e: after_recording_callback(interaction, filename, e))
    client.loop.create_task(stop_listening_after(voice_client, 10.0))


def after_recording_callback(interaction, filename, exception=None):
    if exception:
        print(f"Recording error: {exception}")
        return

    if not interaction: return
    client.loop.call_soon_threadsafe(process_audio_task, interaction, filename)


def process_audio_task(interaction, filename):
    client.loop.create_task(handle_audio_processing(interaction, filename))


async def handle_audio_processing(interaction, filename):
    try:
        print(f"Processing STT for {filename}...")
        stt_recognize = client.loop.run_in_executor(None, transcribe_audio_file, filename)
        transcript = await stt_recognize

        if not transcript:
            await interaction.channel.send("I didn't hear anything. Speak up!")
            return

        await interaction.channel.send(f"You said: \"{transcript}\"")

        # --- GENERATE RESPONSE USING NEW HELPER ---
        print("Sending to Gemini...")
        response = await generate_skippy_response(transcript)

        gemini_response = ""
        if response and response.text:
            gemini_response = response.text
        else:
            gemini_response = "Hmph. The censors silenced my brilliance."
            if response and response.candidates:
                print(f"Safety Block: {response.candidates[0].finish_reason}")

        await interaction.channel.send(f"**Skippy:** {gemini_response}")

        # Clean text for TTS
        tts_text = gemini_response.replace('**', '').replace('*', '').strip()
        if tts_text:
            await speak_text(interaction, tts_text)

    except Exception as e:
        print(f"Error in processing: {e}")
        if not interaction.response.is_done():
            await interaction.followup.send("Something went wrong.")
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass


def transcribe_audio_file(filename: str) -> str:
    try:
        with open(filename, "rb") as audio_file:
            content = audio_file.read()
        with wave.open(filename, "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            audio_channel_count=channels
        )
        response = stt_client.recognize(config=config, audio=audio)
        if response.results:
            return response.results[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"STT Error: {e}")
        return ""


async def speak_text(interaction, text):
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=TTS_VOICE_NAME)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = await client.loop.run_in_executor(
            None,
            tts_client.synthesize_speech,
            texttospeech.SynthesizeSpeechRequest(input=synthesis_input, voice=voice, audio_config=audio_config)
        )

        voice_client = interaction.guild.voice_client
        if not voice_client: return

        temp_audio_file = f"tts_{int(time.time())}.mp3"
        with open(temp_audio_file, "wb") as out:
            out.write(response.audio_content)

        if voice_client.is_playing():
            voice_client.stop()

        voice_client.play(
            discord.FFmpegPCMAudio(temp_audio_file),
            after=lambda e: after_speech_cleanup(e, temp_audio_file)
        )
    except Exception as e:
        print(f"TTS Error: {e}")


def after_speech_cleanup(error, filename):
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except:
            pass


async def send_long_message(interaction, text):
    # (Simplified for brevity, original logic was fine)
    if len(text) <= 2000:
        if not interaction.response.is_done():
            await interaction.followup.send(text)
        else:
            await interaction.channel.send(text)
    else:
        # Basic chunking
        for i in range(0, len(text), 2000):
            await interaction.channel.send(text[i:i + 2000])


# --- 11. Run ---
try:
    client.run(DISCORD_TOKEN)
except Exception as e:
    print(f"Run Error: {e}")