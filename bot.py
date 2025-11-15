import os
import discord
import google.generativeai as genai
from dotenv import load_dotenv

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
        model_name="gemini-1.5-flash-latest", # Using Flash for speed
        generation_config=generation_config,
        system_instruction=system_instruction, # <-- Now using your defined personality
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

# --- 4. Bot Events ---

@client.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""
    print(f'Sprint 1 Bot is online. Logged in as {client.user}')
    # Set a custom status
    await client.change_presence(activity=discord.Game(name="Waiting for !ask"))

@client.event
async def on_message(message):
    """Called every time a message is sent in a channel the bot can see."""
    
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # --- Sprint 1: Text Command (!ask) ---
    if message.content.startswith('!ask'):
        
        # Get the prompt text (everything after "!ask ")
        prompt_text = message.content[5:].strip()

        if not prompt_text:
            await message.reply("Please provide a prompt after `!ask`. For example: `!ask Who are you?`")
            return

        print(f"Received prompt from {message.author}: {prompt_text}")

        # Send a "typing..." indicator to the channel
        async with message.channel.typing():
            try:
                # --- Send to Gemini API ---
                # Using generate_content for a simple request-response
                response = await gemini_model.generate_content_async(prompt_text)
                
                # --- Send Response to Discord ---
                if response and response.text:
                    # Discord has a 2000 character limit per message.
                    # We'll split the message if it's too long.
                    await send_long_message(message.reply, response.text)
                else:
                    await message.reply("Sorry, I couldn't get a response from Gemini.")

            except Exception as e:
                print(f"An error occurred while processing Gemini request: {e}")
                await message.reply(f"An error occurred: {e}")

# --- 5. Helper Function ---
async def send_long_message(send_func, text):
    """
    Splits a long string into 2000-character chunks and sends them.
    `send_func` should be a function like `message.reply` or `channel.send`.
    """
    max_len = 2000
    if len(text) <= max_len:
        await send_func(text)
        return

    # Split into chunks
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    
    # Send the first chunk as a reply, and subsequent chunks as new messages
    if chunks:
        await send_func(chunks[0])
        for chunk in chunks[1:]:
            await send_func.channel.send(chunk)


# --- 6. Run the Bot ---
try:
    client.run(DISCORD_TOKEN)
except discord.errors.LoginFailure:
    print("\n--- LOGIN FAILED ---")
    print("The DISCORD_TOKEN in your .env file is incorrect.")
    print("Please get a new token from the Discord Developer Portal.")
except Exception as e:
    print(f"\nError running bot: {e}")