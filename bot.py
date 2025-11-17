import os
import discord
import google.generativeai as genai
from dotenv import load_dotenv
from discord import app_commands
from google.cloud import texttospeech  # <-- Import is the same

# --- 1. Load Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BOT_PERSONALITY = os.getenv('BOT_PERSONALITY', 'You are a helpful, witty, and concise assistant.')

# This new var tells the Google Cloud library where to find our JSON key
# Make sure the .json file is in your bot's main directory
GOOGLE_SERVICE_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if GOOGLE_SERVICE_JSON:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_SERVICE_JSON
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set. TTS will fail.")

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

    # --- THIS IS THE FIRST CHANGE ---
    # We are switching to the SYNCHRONOUS client to avoid event loop conflicts.
    tts_client = texttospeech.TextToSpeechClient()
    print("Google Cloud TTS client configured successfully.")
except Exception as e:
    print(f"Error configuring Google Cloud TTS: {e}")
    # We don't exit(), as the text bot might still be useful.

# --- 4. Configure Discord Bot ---
# We need to enable all intents to get message content, members, and voice states
intents = discord.Intents.default()
intents.message_content = True  # For text commands (if we used them)
intents.voice_states = True  # For knowing who is in what voice channel
intents.members = True  # For knowing about users

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)  # <-- Create a command tree


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

    # 4. Synthesize Speech
    try:
        print(f"Synthesizing speech for: '{text}'")

        # --- THIS IS THE SECOND CHANGE ---
        # We configure the request objects in the main (async) thread
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Then, we run the BLOCKING tts_client.synthesize_speech call
        # in a separate thread using client.loop.run_in_executor.
        # This prevents it from blocking the bot or conflicting with the event loop.
        response = await client.loop.run_in_executor(
            None,  # Use the default thread pool
            tts_client.synthesize_speech,  # The blocking function to run
            synthesis_input,  # First argument for the function
            voice,  # Second argument
            audio_config  # Third argument
        )
        # The 'await' waits for the thread to finish, then we continue.

        # 5. Save audio to a temporary file
        temp_audio_file = "output.mp3"
        with open(temp_audio_file, "wb") as out:
            out.write(response.audio_content)
        print(f"Audio content written to {temp_audio_file}")

        # 6. Play the audio
        if voice_client.is_playing():
            voice_client.stop()

        audio_source = discord.FFmpegPCMAudio(temp_audio_file)
        voice_client.play(audio_source, after=lambda e: print(f'Finished playing, error: {e}'))

        await interaction.followup.send(f"I said: \"{text}\"")

    except Exception as e:
        print(f"Error during TTS or playback: {e}")
        await interaction.followup.send(f"An error occurred while trying to speak: {e}")


# --- 9. Helper Function ---
async def send_long_message(interaction: discord.Interaction, text: str):
    """Sends a long message, splitting it into chunks if necessary."""
    if len(text) <= 2000:
        await interaction.followup.send(text)
        return

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

    # Send the first chunk as a followup
    if chunks:
        await interaction.followup.send(chunks[0])
        # Send subsequent chunks as new messages in the channel
        if len(chunks) > 1 and interaction.channel:
            for chunk in chunks[1:]:
                await interaction.channel.send(chunk)


# --- 10. Run the Bot ---
try:
    client.run(DISCORD_TOKEN)
except Exception as e:
    print(f"An error occurred while running the bot: {e}")
    print("This might be a simple config error (wrong token) or a network issue.")