import os
import discord
import google.generativeai as genai
from dotenv import load_dotenv
from discord import app_commands  # <-- New import

# --- 1. Load Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# Load the personality, providing a sensible default if not set
BOT_PERSONALITY = os.getenv('BOT_PERSONALITY', 'You are a helpful, witty, and concise assistant.')

if not DISCORD_TOKEN:
    print("Error: DISCORD_TOKEN not found. Make sure to set it in your .env file.")
    exit()

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found. Make sure to set it in your .env file.")
    exit()

# --- 2. Configure Gemini ---
# (We'll keep this simple for Sprint 1. No chat history yet.)
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

# --- 3. Configure Discord Bot ---
# We need to enable all intents to get message content, members, and voice states
intents = discord.Intents.all()

# Set the bot's allowed mentions to none, so it can't @everyone
allowed_mentions = discord.AllowedMentions.none()

client = discord.Client(intents=intents, allowed_mentions=allowed_mentions)
tree = app_commands.CommandTree(client)  # <-- Create a command tree


# --- 4. Bot Events ---

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
    """Called every time a message is sent in a channel the bot can see."""

    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # We no longer process !ask commands here. All logic is in the slash command.


# --- 5. Slash Commands (Text) ---

@tree.command(name="ask", description="Ask a question to the Gemini AI.")
@app_commands.describe(prompt="The question you want to ask.")
async def ask(interaction: discord.Interaction, prompt: str):
    """Handles the /ask slash command."""

    # Defer the response immediately.
    # This shows a "Bot is thinking..." message and prevents a 3-second timeout.
    await interaction.response.defer()

    print(f"Received prompt from {interaction.user}: {prompt}")

    try:
        # --- Send to Gemini API ---
        # Using generate_content for a simple request-response
        response = await gemini_model.generate_content_async(prompt)

        # --- Send Response to Discord ---
        if response and response.text:
            # Use the helper to send the (potentially long) response
            await send_long_message(interaction, response.text)
        else:
            await interaction.followup.send("Sorry, I couldn't get a response from Gemini.")

    except Exception as e:
        print(f"An error occurred while processing Gemini request: {e}")
        await interaction.followup.send(f"An error occurred: {e}")


# --- 6. Slash Commands (Voice) ---

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


# --- 7. Helper Function ---
async def send_long_message(interaction: discord.Interaction, text: str):
    """
    Splits a long string into 2000-character chunks and sends them.
    The first message is sent as a followup, and subsequent messages are sent to the channel.
    """
    max_len = 2000
    if len(text) <= max_len:
        await interaction.followup.send(text)  # Send as the followup
        return

    # Split into chunks
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]

    if chunks:
        # Send the first chunk as the followup
        await interaction.followup.send(chunks[0])

        # Send subsequent chunks as new messages in the same channel
        if len(chunks) > 1 and interaction.channel:
            for chunk in chunks[1:]:
                await interaction.channel.send(chunk)


# --- 8. Run the Bot ---
try:
    client.run(DISCORD_TOKEN)
except discord.errors.LoginFailure:
    print("\n--- LOGIN FAILED ---")
    print("The DISCORD_TOKEN in your .env file is incorrect.")
    print("Please get a new token from the Discord Developer Portal.")
except Exception as e:
    print(f"\nError running bot: {e}")