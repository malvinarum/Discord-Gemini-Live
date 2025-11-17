import os
import discord
import google.generativeai as genai
from dotenv import load_dotenv
from discord import app_commands
from discord import sinks  # <-- Force-importing the sinks module
from google.cloud import texttospeech
from google.cloud import speech  # <-- New import for STT
import wave  # <-- New import for saving audio
import time  # <-- New import for file naming

# --- 1. Load Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BOT_PERSONALITY = os.getenv('BOT_PERSONALITY', 'You are a helpful, witty, and concise assistant.')

# This new var tells the Google Cloud library where to find our JSON key
# Make sure the .json file is in your bot's main directory
GOOGLE_SERVICE_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if GOOGLE_SERVICE_JSON and os.path.exists(GOOGLE_SERVICE_JSON):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_SERVICE_JSON
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set or file not found. TTS/STT will fail.")

# --- 2. Configure Gemini ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Use the personality from the .env file as the system instruction
    system_instruction = BOT_PERSONALITY

    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",  # Using 2.5-flash (stable) instead of 1.5-flash-latest (retired)
        generation_config=generation_config,
        system_instruction=system_instruction,  # <-- Now using your defined personality
    )
    print(f"Gemini model configured successfully with personality: {BOT_PERSONALITY}")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    exit()

# --- 3. Configure Google Cloud TTS ---
try:
    # This client will automatically find and use the credentials
    # we set in os.environ['GOOGLE_APPLICATION_CREDENTIALS']

    # We are using the SYNCHRONOUS client to avoid asyncio event loop conflicts.
    tts_client = texttospeech.TextToSpeechClient()
    print("Google Cloud TTS client configured successfully.")
except Exception as e:
    print(f"Error configuring Google Cloud TTS: {e}")
    # We don't exit(), as the text bot might still be useful.

# --- 3.5. Configure Google Cloud STT ---
try:
    # This client also uses the same GOOGLE_APPLICATION_CREDENTIALS
    stt_client = speech.SpeechClient()
    print("Google Cloud STT client configured successfully.")
except Exception as e:
    print(f"Error configuring Google Cloud STT: {e}")

# --- 4. Configure Discord Bot ---
# We need to enable all intents to get message content, members, and voice states
intents = discord.Intents.default()
intents.message_content = True  # For text commands (if we used them)
intents.voice_states = True  # For knowing who is in what voice channel
intents.members = True  # For knowing about users

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# --- 4.5. Global State ---
# This is a simple way to pass the 'interaction' object
# to our 'after_recording' callback.
# We'll store the interaction, keyed by the filename.
pending_chats = {}


# --- 5. Bot Events ---

@client.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""

    print("Syncing command tree...")
    try:
        # Sync the command tree. This tells Discord about our slash commands.
        # This can take a moment, especially on first run.
        await tree.sync()
        print("Command tree synced.")
    except Exception as e:
        print(f"Error syncing command tree: {e}")

    print(f'Sprint 1 Bot (Slash) is online. Logged in as {client.user}')
    # Set a custom status
    await client.change_presence(activity=discord.Game(name="Waiting for commands..."))


@client.event
async def on_message(message):
    """
    This event is still useful for other things, but we don't
    use it for commands anymore. We'll leave it empty for now.
    """
    # Don't respond to ourselves
    if message.author == client.user:
        return

    # We are all slash commands now, so this function is intentionally blank.


# --- 6. Slash Commands (Text) ---

@tree.command(name="ask", description="Ask a question to the Gemini AI.")
@app_commands.describe(prompt="The question you want to ask.")
async def ask(interaction: discord.Interaction, prompt: str):
    """Handles the /ask slash command."""

    # Send a "thinking..." message
    await interaction.response.defer()

    try:
        print(f"User '{interaction.user.name}' asked: {prompt}")

        # Start a new chat session for this single question
        # We're not storing history *yet*.
        chat_session = gemini_model.start_chat(history=[])

        # Send the prompt to Gemini
        response = await chat_session.send_message_async(prompt)

        # Send the response back to Discord, handling long messages
        await send_long_message(interaction, response.text)

    except Exception as e:
        print(f"An error occurred while processing Gemini request: {e}")
        await interaction.followup.send(f"An error occurred: {e}")


# --- 7. Slash Commands (Voice) ---

@tree.command(name="join", description="Joins your current voice channel.")
async def join(interaction: discord.Interaction):
    """Handles the /join slash command."""

    # Check if the user is in a voice channel
    if not interaction.user.voice:
        await interaction.response.send_message("You're not in a voice channel, meat-bag. Where am I supposed to go?",
                                                ephemeral=True)
        return

    voice_channel = interaction.user.voice.channel

    # Check if the bot is already in a voice channel
    if interaction.guild.voice_client:
        # If already in the user's channel
        if interaction.guild.voice_client.channel == voice_channel:
            await interaction.response.send_message("I'm already *in* your channel. Are you paying attention?",
                                                    ephemeral=True)
            return
        # If in another channel, move to the user's channel
        try:
            await interaction.guild.voice_client.move_to(voice_channel)
            await interaction.response.send_message(f"Fine, I'm moving to `{voice_channel.name}`.")
        except Exception as e:
            await interaction.response.send_message(f"I had a problem moving channels: {e}")
        return

    # If not in any channel, try to connect
    try:
        await voice_channel.connect()
        await interaction.response.send_message(f"Okay, I'm in `{voice_channel.name}`. What do you want?")
    except discord.errors.ClientException as e:
        await interaction.response.send_message(f"Error connecting to voice: {e}", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"An unknown error occurred: {e}", ephemeral=True)


@tree.command(name="leave", description="Leaves the voice channel it is currently in.")
async def leave(interaction: discord.Interaction):
    """Handles the /leave slash command."""

    # Check if the bot is in a voice channel
    if not interaction.guild.voice_client:
        await interaction.response.send_message("I'm not in a voice channel, so I can't leave. Obviously.",
                                                ephemeral=True)
        return

    # If in a voice channel, disconnect
    try:
        current_channel = interaction.guild.voice_client.channel.name
        await interaction.guild.voice_client.disconnect()
        await interaction.response.send_message(
            f"Alright, I'm leaving `{current_channel}`. It was getting boring anyway.")
    except Exception as e:
        await interaction.response.send_message(f"I had a problem leaving: {e}", ephemeral=True)


# --- 8. Slash Command (TTS Test) ---

@tree.command(name="say", description="Speaks the given text in your voice channel.")
@app_commands.describe(text="The text you want the bot to say.")
async def say(interaction: discord.Interaction, text: str):
    """Handles the /say slash command for TTS."""

    # 1. Check if user is in a voice channel
    if not interaction.user.voice:
        await interaction.response.send_message("You need to be in a voice channel for me to speak, meat-bag.",
                                                ephemeral=True)
        return

    voice_channel = interaction.user.voice.channel

    # 2. Defer response
    await interaction.response.defer()

    # 3. Connect to voice
    voice_client = interaction.guild.voice_client
    if not voice_client:
        try:
            voice_client = await voice_channel.connect()
        except Exception as e:
            await interaction.followup.send(f"Failed to connect to voice channel: {e}")
            return
    elif voice_client.channel != voice_channel:
        try:
            await voice_client.move_to(voice_channel)
        except Exception as e:
            await interaction.followup.send(f"Failed to move to your voice channel: {e}")
            return

    # 4. Synthesize Speech & Play
    try:
        # Use our reusable helper function
        await speak_text(voice_client, text)
        await interaction.followup.send(f"I said: \"{text}\"")

    except Exception as e:
        print(f"Error during /say command: {e}")
        await interaction.followup.send(f"An error occurred while trying to speak: {e}")


# --- 9. Slash Command (Core Loop) ---

@tree.command(name="chat", description="Have a one-shot voice conversation with the AI.")
async def chat(interaction: discord.Interaction):
    """
    Handles the full STT -> Gemini -> TTS loop.
    1. Joins VC
    2. Says "I'm listening" (TTS)
    3. Records user's voice
    4. `after_recording` callback is triggered
    """

    # 1. Check if user is in a voice channel
    if not interaction.user.voice:
        await interaction.response.send_message("You need to be in a voice channel to chat, meat-bag.", ephemeral=True)
        return

    voice_channel = interaction.user.voice.channel
    await interaction.response.defer()  # Defer now, we'll follow up

    # 2. Connect to voice
    voice_client = interaction.guild.voice_client
    if not voice_client:
        try:
            voice_client = await voice_channel.connect()
        except Exception as e:
            await interaction.followup.send(f"Failed to connect to voice channel: {e}")
            return
    elif voice_client.channel != voice_channel:
        try:
            await voice_client.move_to(voice_channel)
        except Exception as e:
            await interaction.followup.send(f"Failed to move to your voice channel: {e}")
            return

    # Stop any previous playing/recording
    if voice_client.is_playing():
        voice_client.stop()
    if voice_client.is_listening():
        voice_client.stop_listening()

    # 3. Say "I'm listening"
    try:
        # We need to wait for speech to finish before recording
        # This is a bit advanced, so we'll just add a small delay for now.
        # A proper fix uses a `discord.PCMVolumeTransformer` and `after` callback.
        await speak_text(voice_client, "I'm listening...")
        await interaction.followup.send("I'm listening... (Will stop after 10s of silence)")

        # This is a simple fix: wait until the bot is no longer speaking
        while voice_client.is_playing():
            await discord.utils.sleep_until(time.time() + 0.1)

    except Exception as e:
        await interaction.followup.send(f"Error speaking: {e}")
        return

    # 4. Start recording
    # We create a unique filename and store this interaction
    # so the callback function can find it.
    filename = f"rec_{interaction.id}_{int(time.time())}.wav"
    pending_chats[filename] = interaction  # Store the interaction

    print(f"Starting recording for {filename}")
    voice_client.listen(
        sinks.WaveSink(filename),  # <-- Using the explicit import
        after=after_recording_callback,
        timeout=10.0  # Stop after 10 seconds of silence
    )

    # We already sent the followup, so we're good.


def after_recording_callback(sink: sinks.WaveSink, exception: Exception = None):  # <-- Using the explicit import
    """
    This function is called *after* the recording stops.
    It runs in a separate thread, so we CANNOT use async Discord methods here.
    We must use client.loop.call_soon_threadsafe to schedule our async handler.
    """
    if exception:
        print(f"Error during recording: {exception}")
        return

    filename = sink.filename
    print(f"Recording finished: {filename}")

    # Find the interaction object we stored earlier
    interaction = pending_chats.pop(filename, None)
    if not interaction:
        print(f"Error: Could not find pending interaction for {filename}")
        return

    # This is the *only* safe way to call an async function
    # from this synchronous callback thread.
    client.loop.call_soon_threadsafe(
        process_audio_task,
        interaction,
        filename
    )


def process_audio_task(interaction: discord.Interaction, filename: str):
    """
    A helper function to schedule the async processing task.
    This just creates a new task in the main event loop.
    """
    client.loop.create_task(handle_audio_processing(interaction, filename))


async def handle_audio_processing(interaction: discord.Interaction, filename: str):
    """
    This is the async function that does all the work.
    STT -> Gemini -> TTS -> Play
    """
    try:
        # 1. --- Speech-to-Text (STT) ---
        print(f"Processing STT for {filename}...")

        # We must run this in an executor to avoid blocking
        stt_recognize = client.loop.run_in_executor(
            None,  # Use default thread pool
            transcribe_audio_file,  # Our new blocking function
            filename
        )
        transcript = await stt_recognize

        if not transcript:
            # We already sent "I'm listening...", so we need to use channel.send
            await interaction.channel.send("I didn't hear anything, meat-bag. Try again.")
            os.remove(filename)  # Clean up
            return

        await interaction.channel.send(f"You said: \"{transcript}\"")
        print(f"STT Transcript: {transcript}")

        # 2. --- Gemini ---
        print("Sending transcript to Gemini...")
        chat_session = gemini_model.start_chat(history=[])
        response = await chat_session.send_message_async(transcript)
        gemini_response = response.text

        await interaction.channel.send(f"**Skippy:** {gemini_response}")
        print(f"Gemini Response: {gemini_response}")

        # 3. --- Text-to-Speech (TTS) ---
        print("Synthesizing Gemini response...")
        voice_client = interaction.guild.voice_client
        if not voice_client:
            print("Voice client is gone, cannot speak.")
            return

        await speak_text(voice_client, gemini_response)

    except Exception as e:
        print(f"Error in handle_audio_processing: {e}")
        try:
            await interaction.followup.send(f"An error occurred during processing: {e}", ephemeral=True)
        except discord.errors.InteractionResponded:
            await interaction.channel.send(f"An error occurred during processing: {e}")
    finally:
        # 4. --- Cleanup ---
        if os.path.exists(filename):
            os.remove(filename)
        print(f"Cleaned up {filename}")


def transcribe_audio_file(filename: str) -> str:
    """
    [Blocking Function]
    Sends a local .wav file to Google STT and returns the transcript.
    """
    try:
        with open(filename, "rb") as audio_file:
            content = audio_file.read()

        # Get WAV file sample rate
        with wave.open(filename, "rb") as wf:
            sample_rate = wf.getframerate()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # LINEAR16 is WAV
            sample_rate_hertz=sample_rate,
            language_code="en-US"
        )

        response = stt_client.recognize(config=config, audio=audio)

        if response.results:
            return response.results[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"Error in transcribe_audio_file: {e}")
        return ""


async def speak_text(voice_client: discord.VoiceClient, text: str):
    """
    [Async Function]
    Our helper function from /say, now reusable.
    Synthesizes text and plays it in the given VoiceClient.
    """
    try:
        print(f"Synthesizing speech for: '{text}'")
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        request = texttospeech.SynthesizeSpeechRequest(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        response = await client.loop.run_in_executor(
            None,
            tts_client.synthesize_speech,
            request
        )

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
        os.remove(filename)
        print(f"Cleaned up TTS file: {filename}")


# --- 10. Helper Function ---
async def send_long_message(interaction: discord.Interaction, text: str):
    """Sends a long message, splitting it into chunks if necessary."""

    # Check if the interaction has already been responded to
    if not interaction.is_expired():
        if not interaction.response.is_done():
            # If we haven't responded at all, use followup.send
            try:
                if len(text) <= 2000:
                    await interaction.followup.send(text)
                    return
            except discord.errors.InteractionResponded:
                # This can happen in a race condition
                pass
            except Exception as e:
                print(f"Error sending first followup: {e}")
                # Fallback to channel send
                pass

    # If we are here, either the interaction was already responded to,
    # or the message is > 2000 chars, or an error occurred.

    # Split the text into chunks
    chunks = []
    current_chunk = ""
    for line in text.split('\n'):
        if len(current_chunk) + len(line) + 1 > 2000:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += '\n' + line
    chunks.append(current_chunk)

    # Send the first chunk
    # We try followup.send first, but if that fails, we use channel.send
    if chunks:
        try:
            if not interaction.response.is_done():
                await interaction.followup.send(chunks[0])
                chunks.pop(0)  # Remove the first chunk
            else:
                # Already responded, just use channel.send
                await interaction.channel.send(chunks[0])
                chunks.pop(0)
        except Exception as e:
            print(f"Error sending chunk: {e}. Retrying with channel.send.")
            # Fallback to channel.send
            if interaction.channel:
                try:
                    await interaction.channel.send(chunks[0])
                    chunks.pop(0)  # Remove the first chunk
                except Exception as e2:
                    print(f"Failed to send message to channel: {e2}")

    # Send subsequent chunks as new messages in the channel
    if chunks and interaction.channel:
        for chunk in chunks:
            if chunk:  # ensure chunk is not empty
                await interaction.channel.send(chunk)


# --- 1. Run the Bot ---
try:
    client.run(DISCORD_TOKEN)
except Exception as e:
    print(f"An error occurred while running the bot: {e}")
    print("This might be a simple config error (wrong token) or a network issue.")