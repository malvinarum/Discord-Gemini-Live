import os
import discord
import google.generativeai as genai
from dotenv import load_dotenv
# We no longer need app_commands, discord.Bot handles it.
from google.cloud import texttospeech
from google.cloud import speech
import wave
import time
import asyncio  # <-- Using this for the wait_for_speech_to_finish

# --- 1. Load Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BOT_PERSONALITY = os.getenv('BOT_PERSONALITY', 'You are a helpful, witty, and concise assistant.')

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

    system_instruction = BOT_PERSONALITY

    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
        system_instruction=system_instruction,
    )
    print(f"Gemini model configured successfully with personality: {BOT_PERSONALITY}")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
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
# We're now using discord.Bot, which is the py-cord way.
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True

# We create a Bot instance instead of a Client instance.
bot = discord.Bot(intents=intents)

# We no longer need a CommandTree! bot.slash_command handles it.

# --- 4.5. Global State ---
pending_chats = {}


# --- 5. Bot Events ---

@bot.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""

    # We don't need to sync the tree, py-cord does it on startup.
    # ADDED a version number so we know this file is running.
    print(f'Sprint 1 Bot (v1.9 - SINK/PEP508 FIX) is online. Logged in as {bot.user}')
    await bot.change_presence(activity=discord.Game(name="Waiting for commands..."))


# We don't need on_message, so it's removed.

# --- 6. Slash Commands (Text) ---

# This is the new decorator for py-cord
@bot.slash_command(name="ask", description="Ask a question to the Gemini AI.")
async def ask(ctx: discord.ApplicationContext, prompt: str):
    """Handles the /ask slash command."""

    # py-cord uses ctx (Context) instead of interaction
    # and ctx.defer() instead of interaction.response.defer()
    await ctx.defer()

    try:
        print(f"User '{ctx.author.name}' asked: {prompt}")

        chat_session = gemini_model.start_chat(history=[])
        response = await chat_session.send_message_async(prompt)

        # We pass the context to our helper
        await send_long_message(ctx, response.text)

    except Exception as e:
        print(f"An error occurred while processing Gemini request: {e}")
        # Use followup since we deferred
        await ctx.followup.send(f"An error occurred: {e}", ephemeral=True)


# --- 7. Slash Commands (Voice) ---

@bot.slash_command(name="join", description="Joins your current voice channel.")
async def join(ctx: discord.ApplicationContext):
    """Handles the /join slash command."""

    # *** THIS IS THE FIX FOR THE "Unknown Interaction" ERROR ***
    # We defer immediately to prevent the "Unknown Interaction"
    # error if the firewall (522) causes a timeout.
    await ctx.defer()

    if not ctx.author.voice:
        await ctx.followup.send("You're not in a voice channel, meat-bag. Where am I supposed to go?", ephemeral=True)
        return

    voice_channel = ctx.author.voice.channel

    if ctx.guild.voice_client:
        if ctx.guild.voice_client.channel == voice_channel:
            await ctx.followup.send("I'm already *in* your channel. Are you paying attention?", ephemeral=True)
            return
        try:
            await ctx.guild.voice_client.move_to(voice_channel)
            await ctx.followup.send(f"Fine, I'm moving to `{voice_channel.name}`.")
        except Exception as e:
            await ctx.followup.send(f"I had a problem moving channels: {e}")
        return

    try:
        await voice_channel.connect()
        # Use followup since we deferred
        await ctx.followup.send(f"Okay, I'm in `{voice_channel.name}`. What do you want?")
    except discord.errors.ClientException as e:
        await ctx.followup.send(f"Error connecting to voice: {e}", ephemeral=True)
    except Exception as e:
        # This is where the 522 firewall error will get caught
        print(f"An error occurred during /join: {e}")
        await ctx.followup.send(f"An unknown error occurred (maybe a firewall?): {e}", ephemeral=True)


@bot.slash_command(name="leave", description="Leaves the voice channel it is currently in.")
async def leave(ctx: discord.ApplicationContext):
    """Handles the /leave slash command."""

    # This command is fast, so defer() isn't strictly needed.
    # But we add it for consistency.
    await ctx.defer(ephemeral=True)

    if not ctx.guild.voice_client:
        await ctx.followup.send("I'm not in a voice channel, so I can't leave. Obviously.")
        return

    try:
        current_channel = ctx.guild.voice_client.channel.name
        await ctx.guild.voice_client.disconnect()
        await ctx.followup.send(f"Alright, I'm leaving `{current_channel}`. It was getting boring anyway.")
    except Exception as e:
        await ctx.followup.send(f"I had a problem leaving: {e}")


# --- 8. Slash Command (TTS Test) ---

@bot.slash_command(name="say", description="Speaks the given text in your voice channel.")
async def say(ctx: discord.ApplicationContext, text: str):
    """Handles the /say slash command for TTS."""

    if not ctx.author.voice:
        await ctx.respond("You need to be in a voice channel for me to speak, meat-bag.", ephemeral=True)
        return

    voice_channel = ctx.author.voice.channel
    await ctx.defer()

    voice_client = ctx.guild.voice_client
    if not voice_client:
        try:
            voice_client = await voice_channel.connect()
        except Exception as e:
            await ctx.followup.send(f"Failed to connect to voice channel (firewall?): {e}")
            return
    elif voice_client.channel != voice_channel:
        try:
            await voice_client.move_to(voice_channel)
        except Exception as e:
            await ctx.followup.send(f"Failed to move to your voice channel: {e}")
            return

    try:
        # Pass the context (ctx) to the helper
        await speak_text(ctx, text)
        await ctx.followup.send(f"I said: \"{text}\"")

    except Exception as e:
        print(f"Error during /say command: {e}")
        await ctx.followup.send(f"An error occurred while trying to speak: {e}")


# --- 9. Slash Command (Core Loop) ---

@bot.slash_command(name="chat", description="Have a one-shot voice conversation with the AI.")
async def chat(ctx: discord.ApplicationContext):
    """
    Handles the full STT -> Gemini -> TTS loop.
    """
    if not ctx.author.voice:
        await ctx.respond("You need to be in a voice channel to chat, meat-bag.", ephemeral=True)
        return

    voice_channel = ctx.author.voice.channel
    await ctx.defer()

    voice_client = ctx.guild.voice_client
    if not voice_client:
        try:
            voice_client = await voice_channel.connect()
        except Exception as e:
            await ctx.followup.send(f"Failed to connect to voice channel (firewall?): {e}")
            return
    elif voice_client.channel != voice_channel:
        try:
            await voice_client.move_to(voice_channel)
        except Exception as e:
            await ctx.followup.send(f"Failed to move to your voice channel: {e}")
            return

    if voice_client.is_playing():
        voice_client.stop()

    # We now assume .stop_listening() exists because py-cord[sink] is installed
    voice_client.stop_listening()

    try:
        await speak_text(ctx, "I'm listening...")
        # Send this *after* speaking, as followup can only be used once.
        await ctx.followup.send("I'm listening... (Will stop after 10s of silence)")

        # Wait for speech to finish before listening
        while voice_client.is_playing():
            await asyncio.sleep(0.1)

    except Exception as e:
        # Check if we already sent the followup
        if not ctx.interaction.response.is_done():
            await ctx.followup.send(f"Error speaking: {e}")
        else:
            await ctx.channel.send(f"Error speaking: {e}")
        return

    filename = f"rec_{ctx.interaction.id}_{int(time.time())}.wav"
    pending_chats[filename] = ctx  # Store the context

    print(f"Starting recording for {filename}")

    # We now assume .listen() exists because py-cord[sink] is installed
    voice_client.listen(
        discord.sinks.WaveSink(filename),
        after=after_recording_callback,
        timeout=10.0
    )


def after_recording_callback(sink: discord.sinks.WaveSink, exception: Exception = None):
    """
    This function is called *after* the recording stops.
    It runs in a separate thread, so we CANNOT use async Discord methods here.
    """
    if exception:
        print(f"Error during recording: {exception}")
        return

    filename = sink.filename
    print(f"Recording finished: {filename}")

    ctx = pending_chats.pop(filename, None)
    if not ctx:
        print(f"Error: Could not find pending context for {filename}")
        return

    # Use bot.loop (from our Bot object) to schedule the async task
    bot.loop.call_soon_threadsafe(
        process_audio_task,
        ctx,
        filename
    )


def process_audio_task(ctx: discord.ApplicationContext, filename: str):
    """
    A helper function to schedule the async processing task.
    """
    bot.loop.create_task(handle_audio_processing(ctx, filename))


async def handle_audio_processing(ctx: discord.ApplicationContext, filename: str):
    """
    This is the async function that does all the work.
    STT -> Gemini -> TTS -> Play
    """
    try:
        print(f"Processing STT for {filename}...")

        stt_recognize = bot.loop.run_in_executor(
            None,
            transcribe_audio_file,
            filename
        )
        transcript = await stt_recognize

        if not transcript:
            await ctx.channel.send("I didn't hear anything, meat-bag. Try again.")
            os.remove(filename)
            return

        await ctx.channel.send(f"You said: \"{transcript}\"")
        print(f"STT Transcript: {transcript}")

        print("Sending transcript to Gemini...")
        chat_session = gemini_model.start_chat(history=[])
        response = await chat_session.send_message_async(transcript)
        gemini_response = response.text

        await ctx.channel.send(f"**Skippy:** {gemini_response}")
        print(f"Gemini Response: {gemini_response}")

        print("Synthesizing Gemini response...")

        await speak_text(ctx, gemini_response)

    except Exception as e:
        print(f"Error in handle_audio_processing: {e}")
        try:
            # Check if we already sent the followup
            if not ctx.interaction.response.is_done():
                await ctx.followup.send(f"An error occurred during processing: {e}", ephemeral=True)
            else:
                await ctx.channel.send(f"An error occurred during processing: {e}")
        except discord.errors.InteractionResponded:
            await ctx.channel.send(f"An error occurred during processing: {e}")
    finally:
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

        with wave.open(filename, "rb") as wf:
            sample_rate = wf.getframerate()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
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


async def speak_text(ctx: discord.ApplicationContext, text: str):
    """
    [Async Function]
    Our helper function from /say, now reusable.
    Re-fetches the voice_client to prevent "Not connected" errors.
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

        # Use bot.loop for the executor
        response = await bot.loop.run_in_executor(
            None,
            tts_client.synthesize_speech,
            request
        )

        # --- FIX: Re-fetch the voice client ---
        # The old voice_client object might be stale after the await.
        voice_client = ctx.guild.voice_client
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
        os.remove(filename)
        print(f"Cleaned up TTS file: {filename}")


# --- 10. Helper Function ---
async def send_long_message(ctx: discord.ApplicationContext, text: str):
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
        if not ctx.interaction.response.is_done():
            await ctx.followup.send(first_chunk)
        else:
            await ctx.channel.send(first_chunk)

        # Send subsequent chunks as new messages in the channel
        for chunk in chunks:
            if chunk:  # ensure chunk is not empty
                await ctx.channel.send(chunk)

    except discord.errors.InteractionResponded:
        # If we already responded (e.g., in a race condition)
        # just send all chunks to the channel
        await ctx.channel.send(first_chunk)
        for chunk in chunks:
            if chunk:
                await ctx.channel.send(chunk)
    except Exception as e:
        print(f"Error in send_long_message: {e}")
        # Fallback for safety
        await ctx.channel.send("An error occurred while splitting the message.")


# --- 11. Run the Bot ---
# THIS IS THE LINE THAT WAS MISSING
try:
    bot.run(DISCORD_TOKEN)  # Use bot.run, not client.run
except Exception as e:
    print(f"An error occurred while running the bot: {e}")
    print("This might be a simple config error (wrong token) or a network issue.")