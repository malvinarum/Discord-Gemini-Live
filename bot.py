import os
import discord
from discord import app_commands
from google import genai
from google.genai import types
from dotenv import load_dotenv
from google.cloud import texttospeech
# Removed: from google.cloud import speech (We don't need STT anymore!)
import wave
import time
import asyncio
import functools
from discord.ext import voice_recv

# --- 1. Load Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BOT_PERSONALITY = os.getenv('BOT_PERSONALITY', 'You are a helpful, witty, and concise assistant.')

# --- VOICE CONFIGURATION UPDATE ---
# Switched to 'en-GB-Neural2-D': A deep, British male voice (Neural2 is more human-like than WaveNet)
TTS_VOICE_NAME = os.getenv('TTS_VOICE_NAME', 'en-GB-Neural2-D')

# --- MODEL CONFIGURATION ---
# We use Flash because it handles Audio Input very quickly and cheaply
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

GOOGLE_SERVICE_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if GOOGLE_SERVICE_JSON and os.path.exists(GOOGLE_SERVICE_JSON):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_SERVICE_JSON
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set or file not found. TTS/STT will fail.")

# --- 2. Configure Gemini (New SDK Pattern) ---
try:
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

    print(f"Gemini Client initialized. Target Model: {GEMINI_MODEL_NAME}")

except Exception as e:
    print(f"FATAL Error configuring Gemini Client: {e}")
    exit()

# --- 3. Configure Google Cloud TTS ---
try:
    tts_client = texttospeech.TextToSpeechClient()
    print("Google Cloud TTS client configured successfully.")
except Exception as e:
    print(f"Error configuring Google Cloud TTS: {e}")

# --- 4. Configure Discord Bot ---
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# --- HELPER: Run Sync Gemini Calls in Thread ---
async def generate_skippy_response(prompt_input):
    """
    Wraps the synchronous Gemini Client call.
    'prompt_input' can now be text OR an audio Part object.
    """

    def _call_gemini():
        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries):
            try:
                response = client_genai.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=[prompt_input],  # Pass as a list
                    config=generation_config
                )
                return response
            except Exception as inner_e:
                error_str = str(inner_e)
                if "503" in error_str or "429" in error_str or "overloaded" in error_str.lower():
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        print(f"Gemini overloaded (Attempt {attempt + 1}/{max_retries}). Retrying...")
                        time.sleep(sleep_time)
                        continue
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

    print(f'Skippy (Native Audio Edition) is online. Logged in as {client.user}')
    await client.change_presence(activity=discord.Game(name="Listening to your nonsense..."))


# --- 6. Slash Commands (Text) ---

@tree.command(name="ask", description="Ask a question to the Gemini AI.")
async def ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()

    try:
        print(f"User '{interaction.user.name}' asked: {prompt}")
        # For text, we just pass the string
        response = await generate_skippy_response(prompt)

        gemini_response = ""
        if response and response.text:
            gemini_response = response.text
        else:
            gemini_response = "Hmph. The cosmic censors blocked me."

        await send_long_message(interaction, f"**Skippy:** {gemini_response}")

    except Exception as e:
        print(f"An error occurred: {e}")
        await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)


# --- 7. Slash Commands (Voice Join/Leave/Say) ---
# (These remain largely the same, ensuring we have voice_recv)

@tree.command(name="join", description="Joins your current voice channel.")
async def join(interaction: discord.Interaction):
    await interaction.response.defer()
    if not interaction.user.voice:
        await interaction.followup.send("You're not in a voice channel.", ephemeral=True)
        return
    voice_channel = interaction.user.voice.channel

    if interaction.guild.voice_client:
        await interaction.followup.send("I'm already connected.")
        return

    try:
        await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        await interaction.followup.send(f"Connected to `{voice_channel.name}`.")
    except Exception as e:
        await interaction.followup.send(f"Error connecting: {e}", ephemeral=True)


@tree.command(name="leave", description="Leaves the voice channel.")
async def leave(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    if not interaction.guild.voice_client:
        await interaction.followup.send("I'm not connected.")
        return
    await interaction.guild.voice_client.disconnect()
    await interaction.followup.send("Disconnected.")


@tree.command(name="say", description="Speaks the given text.")
async def say(interaction: discord.Interaction, text: str):
    if not interaction.user.voice:
        await interaction.response.send_message("Join a voice channel.", ephemeral=True)
        return
    await interaction.response.defer()

    voice_client = interaction.guild.voice_client
    if not voice_client:
        try:
            voice_client = await interaction.user.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        except Exception as e:
            await interaction.followup.send(f"Failed to connect: {e}")
            return

    await speak_text(interaction, text)
    await interaction.followup.send(f"Spoke: {text}")


# --- 8. Slash Command (Chat - Native Audio) ---

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
        print(f"Processing Native Audio for {filename}...")

        # 1. Read the audio file bytes
        with open(filename, "rb") as f:
            audio_bytes = f.read()

        # 2. Prepare the input Part for Gemini
        # This replaces the STT text string. We send the raw audio.
        prompt_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        )

        # 3. Send to Gemini (Native Audio Input)
        print("Sending AUDIO bytes directly to Gemini...")
        response = await generate_skippy_response(prompt_part)

        gemini_response = ""
        if response and response.text:
            gemini_response = response.text
            print(f"Skippy heard you and said: {gemini_response}")
        else:
            gemini_response = "Hmph. I heard noises, but the censors blocked my witty retort."
            if response and response.candidates:
                print(f"Safety Block: {response.candidates[0].finish_reason}")

        await interaction.channel.send(f"**Skippy:** {gemini_response}")

        tts_text = gemini_response.replace('**', '').replace('*', '').strip()
        if tts_text:
            await speak_text(interaction, tts_text)

    except Exception as e:
        print(f"Error in processing: {e}")
        if not interaction.response.is_done():
            await interaction.followup.send("Something went wrong processing the audio.")
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass


async def speak_text(interaction, text):
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Use the NEW voice name
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB",  # British for "Poppycock!"
            name=TTS_VOICE_NAME  # en-GB-Neural2-D
        )

        # Tweak the audio to sound more "tired wizard"
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.85,  # Slower (Exhausted)
            pitch=-4.0  # Deeper (Ancient)
        )

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
    if len(text) <= 2000:
        if not interaction.response.is_done():
            await interaction.followup.send(text)
        else:
            await interaction.channel.send(text)
    else:
        for i in range(0, len(text), 2000):
            await interaction.channel.send(text[i:i + 2000])


# --- 11. Run ---
try:
    client.run(DISCORD_TOKEN)
except Exception as e:
    print(f"Run Error: {e}")