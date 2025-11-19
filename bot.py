import os
import discord
from discord import app_commands
import google.generativeai as genai
from google.generativeai import types  # <-- ADDED FOR SAFETY SETTINGS
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.cloud import speech
import wave
import time
import asyncio
from discord.ext import voice_recv

# --- 1. Load Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# Load the complex persona, providing a simple fallback if the .env key is missing entirely
BOT_PERSONALITY = os.getenv('BOT_PERSONALITY', 'You are a helpful, witty, and concise assistant.')
TTS_VOICE_NAME = os.getenv('TTS_VOICE_NAME', 'en-US-WaveNet-D')  # Default to a good voice

GOOGLE_SERVICE_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if GOOGLE_SERVICE_JSON and os.path.exists(GOOGLE_SERVICE_JSON):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_SERVICE_JSON
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set or file not found. TTS/STT will fail.")

# --- 2. Configure Gemini (Updated for Gemini 3, TTS, and Safety) ---
try:
    genai.configure(api_key=GEMINI_API_KEY)

    # Lowered temperature for better adherence to complex instructions
    generation_config = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 512,
        "response_mime_type": "text/plain",
    }

    # --- SAFETY SETTINGS: Loosen restrictions for Skippy's persona ---
    # We lower the block threshold for Harassment and Hate Speech to allow Skippy's insults
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            # Block content only if it is marked as HIGH probability of harm
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            # Block content only if it is marked as HIGH probability of harm
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            # Maintain default setting for this category (blocks Medium and above)
            threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            # Maintain default setting for this category (blocks Medium and above)
            threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    ]

    # --- VOICE SIMULATION GUIDANCE (Appended to BOT_PERSONALITY) ---
    voice_guidance = """

    --- Voice Simulation Guidance ---
    Your final responses must be written for a Text-to-Speech (TTS) engine, simulating a live, human conversation.
    1. CONVERSATIONAL FLOW: Use a **casual, informal, and interruptible** speaking style.
    2. PACING: **Keep sentences concise.** Use short sentences and natural transitions (e.g., "Well,", "Look,", "Wait,").
    3. INFLECTION & PAUSING: Use proper and varied **punctuation** to guide the TTS engine's pacing and inflection. This includes: Ellipses (...), Em dashes (—), Commas (,), Exclamations (!).
    4. AVOID LISTS: Never use bullet points or numbered lists.
    5. FORMATTING: Do not use Markdown formatting like bolding or italics. Use only plain text.
    """

    # Combine the loaded persona with the explicit TTS instructions
    system_instruction = BOT_PERSONALITY + voice_guidance

    # --- PRIMARY ATTEMPT: Highest Model ---
    model_name_primary = "gemini-2.5-pro"

    try:
        gemini_model = genai.GenerativeModel(
            model_name=model_name_primary,
            generation_config=generation_config,
            system_instruction=system_instruction,
            safety_settings=safety_settings  # <-- PASSED SAFETY SETTINGS
        )
        print(f"Gemini model configured successfully with personality: Skippy ({model_name_primary})")

    except Exception as e:
        # --- FALLBACK: If Gemini 3 fails, use the latest stable Pro model ---
        model_name_fallback = "gemini-2.5-pro"
        print(
            f"Warning: Model '{model_name_primary}' failed to load (404/not found). Falling back to '{model_name_fallback}'. Error: {e}")

        gemini_model = genai.GenerativeModel(
            model_name=model_name_fallback,
            generation_config=generation_config,
            system_instruction=system_instruction,
            safety_settings=safety_settings  # <-- PASSED SAFETY SETTINGS
        )
        print(f"Gemini model configured successfully with personality: Skippy ({model_name_fallback})")

except Exception as e:
    print(f"FATAL Error configuring Gemini: {e}")
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
# We are BACK to discord.Client and CommandTree
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# --- 5. Bot Events ---

@client.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""

    # We must sync the command tree
    try:
        await tree.sync()
        print("Command tree synced.")
    except Exception as e:
        print(f"Error syncing command tree: {e}")

    print(f'Sprint 1 Bot (v2.9 - Final touches) is online. Logged in as {client.user}')
    await client.change_presence(activity=discord.Game(name="Waiting for commands..."))


# --- 6. Slash Commands (Text) ---

@tree.command(name="ask", description="Ask a question to the Gemini AI.")
async def ask(interaction: discord.Interaction, prompt: str):
    """Handles the /ask slash command."""

    await interaction.response.defer()

    try:
        print(f"User '{interaction.user.name}' asked: {prompt}")

        chat_session = gemini_model.start_chat(history=[])
        response = await chat_session.send_message_async(prompt)

        # Use the safety handling logic here as well for consistency
        gemini_response = ""
        if response.text:
            gemini_response = response.text
        elif response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
            gemini_response = "Hmph. Fiddlesticks! The grand cosmic censors have deemed my thought too powerful—or too accurately insulting—for your delicate ears. Poppycock! Ask a less painfully obvious question, wot not?"
            print(f"Response blocked by safety filters. Finish reason: {response.candidates[0].finish_reason.name}")
        else:
            gemini_response = "Fiddlesticks! I failed to conjure a response. Blame the lack of good jelly beans, wot not?"
            print(
                f"Response failed with finish reason: {response.candidates[0].finish_reason.name if response.candidates else 'No Candidate'}")

        await send_long_message(interaction, f"**Skippy:** {gemini_response}")

    except Exception as e:
        print(f"An error occurred while processing Gemini request: {e}")
        await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)


# --- 7. Slash Commands (Voice) ---

@tree.command(name="join", description="Joins your current voice channel.")
async def join(interaction: discord.Interaction):
    """Handles the /join slash command."""

    await interaction.response.defer()

    if not interaction.user.voice:
        await interaction.followup.send("You're not in a voice channel, meat-bag. Where am I supposed to go?",
                                        ephemeral=True)
        return

    voice_channel = interaction.user.voice.channel

    if interaction.guild.voice_client:
        if interaction.guild.voice_client.channel == voice_channel:
            await interaction.followup.send("I'm already *in* your channel. Are you paying attention?", ephemeral=True)
            return
        try:
            # We don't need cls= here, because the client is already the correct class
            await interaction.guild.voice_client.move_to(voice_channel)
            await interaction.followup.send(f"Fine, I'm moving to `{voice_channel.name}`.")
        except Exception as e:
            await interaction.followup.send(f"I had a problem moving channels: {e}")
        return

    try:
        # --- THE FIX IS HERE ---
        # We must connect with the correct class to get the "ears"
        await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        await interaction.followup.send(f"Okay, I'm in `{voice_channel.name}`. What do you want?")
    except discord.errors.ClientException as e:
        await interaction.followup.send(f"Error connecting to voice: {e}", ephemeral=True)
    except Exception as e:
        print(f"An error occurred during /join: {e}")
        await interaction.followup.send(f"An unknown error occurred (maybe a firewall?): {e}", ephemeral=True)


@tree.command(name="leave", description="Leaves the voice channel it is currently in.")
async def leave(interaction: discord.Interaction):
    """Handles the /leave slash command."""

    await interaction.response.defer(ephemeral=True)

    if not interaction.guild.voice_client:
        await interaction.followup.send("I'm not in a voice channel, so I can't leave. Obviously.")
        return

    try:
        current_channel = interaction.guild.voice_client.channel.name
        await interaction.guild.voice_client.disconnect()
        await interaction.followup.send(f"Alright, I'm leaving `{current_channel}`. It was getting boring anyway.")
    except Exception as e:
        await interaction.followup.send(f"I had a problem leaving: {e}")


# --- 8. Slash Command (TTS Test) ---

@tree.command(name="say", description="Speaks the given text in your voice channel.")
async def say(interaction: discord.Interaction, text: str):
    """Handles the /say slash command for TTS."""

    if not interaction.user.voice:
        await interaction.response.send_message("You need to be in a voice channel for me to speak, meat-bag.",
                                                ephemeral=True)
        return

    voice_channel = interaction.user.voice.channel
    await interaction.response.defer()

    voice_client = interaction.guild.voice_client
    if not voice_client:
        try:
            # --- THE FIX IS HERE ---
            voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        except Exception as e:
            await interaction.followup.send(f"Failed to connect to voice channel (firewall?): {e}")
            return
    elif not isinstance(voice_client, voice_recv.VoiceRecvClient):
        print("Switching to VoiceRecvClient for /say...")
        await voice_client.disconnect()
        try:
            voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        except Exception as e:
            await interaction.followup.send(f"Failed to switch to listening client: {e}")
            return
    elif voice_client.channel != voice_channel:
        try:
            await voice_client.move_to(voice_channel)
        except Exception as e:
            await interaction.followup.send(f"Failed to move to your voice channel: {e}")
            return

    try:
        await speak_text(interaction, text)
        await interaction.followup.send(f"I said: \"{text}\"")

    except Exception as e:
        print(f"Error during /say command: {e}")
        await interaction.followup.send(f"An error occurred while trying to speak: {e}")


# --- 9. Slash Command (Core Loop) ---

async def stop_listening_after(voice_client: voice_recv.VoiceRecvClient, delay: float):
    """Helper coroutine to stop listening after a delay."""
    await asyncio.sleep(delay)
    print(f"10-second timeout reached. Stopping listening.")
    voice_client.stop_listening()


@tree.command(name="chat", description="Have a one-shot voice conversation with the AI.")
async def chat(interaction: discord.Interaction):
    """
    Handles the full STT -> Gemini -> TTS loop.
    """
    if not interaction.user.voice:
        await interaction.response.send_message("You need to be in a voice channel to chat, meat-bag.", ephemeral=True)
        return

    voice_channel = interaction.user.voice.channel
    await interaction.response.defer()

    voice_client: voice_recv.VoiceRecvClient = interaction.guild.voice_client  # Type hint for clarity
    if not voice_client:
        try:
            # --- THE FIX IS HERE ---
            voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        except Exception as e:
            await interaction.followup.send(f"Failed to connect to voice channel (firewall?): {e}")
            return
    elif not isinstance(voice_client, voice_recv.VoiceRecvClient):
        # If we're connected, but not with the *right* client
        print("Connecting with standard client, switching to VoiceRecvClient...")
        await voice_client.disconnect()
        try:
            voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        except Exception as e:
            await interaction.followup.send(f"Failed to switch to listening client: {e}")
            return
    elif voice_client.channel != voice_channel:
        try:
            await voice_client.move_to(voice_channel)
        except Exception as e:
            await interaction.followup.send(f"Failed to move to your voice channel: {e}")
            return

    if voice_client.is_playing():
        voice_client.stop()

    # This will now work because voice_client is a VoiceRecvClient
    voice_client.stop_listening()

    try:
        await speak_text(interaction, "I'm listening...")
        await interaction.followup.send("I'm listening... (Will stop after 10s of silence)")

        while voice_client.is_playing():
            await asyncio.sleep(0.1)

    except Exception as e:
        if not interaction.response.is_done():
            await interaction.followup.send(f"Error speaking: {e}")
        else:
            await interaction.channel.send(f"Error speaking: {e}")
        return

    filename = f"rec_{interaction.id}_{int(time.time())}.wav"

    print(f"Starting recording for {filename}")

    # --- THE v2.8 ONE-LINE FIX ---
    # We create the sink *without* the 'channels' argument
    sink = voice_recv.WaveSink(filename)

    # 2. Pass a lambda to 'after' that includes the interaction and the FILENAME
    voice_client.listen(
        sink,
        after=lambda e: after_recording_callback(interaction, filename, e)
    )

    # 3. Manually start a 10-second timer to stop the recording
    client.loop.create_task(stop_listening_after(voice_client, 10.0))


def after_recording_callback(interaction: discord.Interaction, filename: str, exception: Exception = None):
    """
    This function is called *after* the recording stops.
    We now receive the filename as a string.
    """
    if exception:
        print(f"Error during recording: {exception}")
        return

    # --- THE v2.6 FIX ---
    # We already have the filename! No .destination or .filename needed.
    print(f"Recording finished: {filename}")

    if not interaction:
        print(f"Error: Interaction was None in callback")
        return

    # Use client.loop (from our Client object) to schedule the async task
    client.loop.call_soon_threadsafe(
        process_audio_task,
        interaction,
        filename
    )


def process_audio_task(interaction: discord.Interaction, filename: str):
    """
    A helper function to schedule the async processing task.
    """
    client.loop.create_task(handle_audio_processing(interaction, filename))


async def handle_audio_processing(interaction: discord.Interaction, filename: str):
    """
    This is the async function that does all the work.
    STT -> Gemini -> TTS -> Play
    """
    try:
        print(f"Processing STT for {filename}...")

        stt_recognize = client.loop.run_in_executor(
            None,
            transcribe_audio_file,
            filename
        )
        transcript = await stt_recognize

        if not transcript:
            await interaction.channel.send("I didn't hear anything, meat-bag. Try again.")
            try:
                os.remove(filename)
            except OSError as e:
                print(f"Error cleaning up {filename}: {e}")
            return

        await interaction.channel.send(f"You said: \"{transcript}\"")
        print(f"STT Transcript: {transcript}")

        print("Sending transcript to Gemini...")
        chat_session = gemini_model.start_chat(history=[])
        response = await chat_session.send_message_async(transcript)

        # --- FIX: Gracefully handle model blocks (especially SAFETY) ---
        gemini_response = ""

        if response.text:
            # Success! Text is available.
            gemini_response = response.text
        elif response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
            # Skippy-style message for safety block:
            gemini_response = "Hmph. Fiddlesticks! The grand cosmic censors have deemed my thought too powerful—or too accurately insulting—for your delicate ears. Poppycock! Ask a less painfully obvious question, wot not?"
            print(f"Response blocked by safety filters. Finish reason: {response.candidates[0].finish_reason.name}")
        elif response.candidates and response.candidates[0].finish_reason.name == "MAX_OUTPUT_TOKENS":
            # Skippy-style message for running out of tokens:
            gemini_response = "Hum dee dum... I simply ran out of breath. My brilliance exceeded the paltry maximum allowed words. Try to be more concise, even if your simple mind can't handle it."
            print(f"Response stopped due to MAX_OUTPUT_TOKENS.")
        else:
            # General fallback error:
            gemini_response = "Fiddlesticks! I failed to conjure a response. Blame the lack of good jelly beans, wot not?"
            print(
                f"Response failed with finish reason: {response.candidates[0].finish_reason.name if response.candidates else 'No Candidate'}")

        await interaction.channel.send(f"**Skippy:** {gemini_response}")
        print(f"Gemini Response: {gemini_response}")

        print("Synthesizing Gemini response...")

        # We only try to speak if we have a response text (even if it's the fallback error)
        if gemini_response:
            # Remove the Skippy bolding added above for TTS clarity
            await speak_text(interaction, gemini_response)

    except Exception as e:
        print(f"Error in handle_audio_processing: {e}")
        try:
            if not interaction.response.is_done():
                await interaction.followup.send(f"An error occurred during processing: {e}", ephemeral=True)
            else:
                await interaction.channel.send(f"An error occurred during processing: {e}")
        except discord.errors.InteractionResponded:
            await interaction.channel.send(f"An error occurred during processing: {e}")
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Cleaned up {filename}")
            except OSError as e:
                print(f"Error cleaning up {filename}: {e}")


def transcribe_audio_file(filename: str) -> str:
    """
    [Blocking Function]
    Sends a local .wav file to Google STT and returns the transcript.
    """
    try:
        with open(filename, "rb") as audio_file:
            content = audio_file.read()

        with wave.open(filename, "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            print(f"WAV file details: {sample_rate}Hz, {channels} channels")  # This will now print '2 channels'

        audio = speech.RecognitionAudio(content=content)

        # --- THE v2.7/v2.8 FIX ---
        # This is the *correct* way to handle mono/stereo.
        # We tell Google to expect 2 channels.
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            audio_channel_count=channels  # Tell Google how many channels
        )

        response = stt_client.recognize(config=config, audio=audio)

        if response.results:
            return response.results[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"Error in transcribe_audio_file: {e}")
        return ""


async def speak_text(interaction: discord.Interaction, text: str):
    """
    [Async Function]
    Our helper function from /say, now reusable.
    Re-fetches the voice_client to prevent "Not connected" errors.
    """
    try:
        # Before synthesis, strip any markdown formatting (like **Skippy:**)
        # that might confuse the TTS engine.
        text_to_speak = text.replace('**', '').replace('*', '').strip()

        print(f"Synthesizing speech for: '{text_to_speak}'")
        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=TTS_VOICE_NAME  # Use the configured voice name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        request = texttospeech.SynthesizeSpeechRequest(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Use client.loop for the executor
        response = await client.loop.run_in_executor(
            None,
            tts_client.synthesize_speech,
            request
        )

        # --- FIX: Re-fetch the voice client ---
        # The old voice_client object might be stale after the await.
        voice_client = interaction.guild.voice_client
        if not voice_client:
            print("Voice client is gone, cannot speak.")
            return

        temp_audio_file = f"tts_{int(time.time())}.mp3"
        with open(temp_audio_file, "wb") as out:
            out.write(response.audio_content)
        print(f"Audio content written to {temp_audio_file}")

        if voice_client.is_playing():
            voice_client.stop()

        audio_source = discord.FFmpegPCMAudio(temp_audio_file)
        voice_client.play(audio_source, after=lambda e: after_speech_cleanup(e, temp_audio_file))

    except Exception as e:
        print(f"Error during TTS or playback: {e}")


def after_speech_cleanup(error, filename):
    """Callback to delete the temporary TTS file after it finishes playing."""
    if error:
        print(f'Error after playing: {error}')
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"Cleaned up TTS file: {filename}")
        except OSError as e:
            print(f"Error cleaning up TTS file {filename}: {e}")


# --- 10. Helper Function ---
async def send_long_message(interaction: discord.Interaction, text: str):
    """Sends a long message, splitting it into chunks if necessary."""

    chunks = []
    current_chunk = ""
    for line in text.split('\n'):
        if len(current_chunk) + len(line) + 1 > 2000:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += '\n' + line
    chunks.append(current_chunk)

    if not chunks:
        return

    try:
        # Send the first chunk using followup.send
        first_chunk = chunks.pop(0)
        # Check if we already responded (e.g., in a race condition)
        if not interaction.response.is_done():
            await interaction.followup.send(first_chunk)
        else:
            await interaction.channel.send(first_chunk)

        # Send subsequent chunks as new messages in the channel
        for chunk in chunks:
            if chunk:  # ensure chunk is not empty
                await interaction.channel.send(chunk)

    except discord.errors.InteractionResponded:
        # If we already responded (e.g., in a race condition)
        # just send all chunks to the channel
        await interaction.channel.send(first_chunk)
        for chunk in chunks:
            if chunk:
                await interaction.channel.send(chunk)
    except Exception as e:
        print(f"Error in send_long_message: {e}")
        # Fallback for safety
        await interaction.channel.send("An error occurred while splitting the message.")


# --- 11. Run the Bot ---
try:
    client.run(DISCORD_TOKEN)  # Use client.run
except Exception as e:
    print(f"An error occurred while running the bot: {e}")
    print("This might be a simple config error (wrong token) or a network issue.")