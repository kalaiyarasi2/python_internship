# Optimized Terminal Interface for Gigi Personal Growth Coach
# Now with multi-user support and long-term memory

import asyncio
import sys
from datetime import datetime
import argparse

# Import the optimized agent
from core import GigiAPI

class TerminalGigiAgent:
    def __init__(self, username: str):
        """Initialize the terminal-based agent"""
        self.api = GigiAPI()
        self.user_id = username
        self.conversation_count = 0

    def print_welcome(self):
        """Display welcome message"""
        print("=" * 50)
        print(f"GIGI - Welcome, {self.user_id}!")
        print("=" * 50)
        print("I'm Gigi, your personal growth coach with long-term memory.")
        print("Type your message to start, or 'help' for commands.")
        print("=" * 50)

    async def get_conversation_history(self):
        """Retrieve and display conversation history for the user"""
        try:
            print("Loading your conversation history...")
            history = await self.api.get_history(self.user_id)
            
            if history['success'] and history.get('conversation_history'):
                print(f"\nRecent Interactions (Total: {len(history['conversation_history'])})")
                print("-" * 60)
                for i, convo in enumerate(history['conversation_history'], 1):
                    timestamp = convo.get('timestamp', 'Unknown')
                    message = convo.get('message', 'N/A')[:70]
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    print(f"{i}. [{dt.strftime('%Y-%m-%d %H:%M')}] {message}...")
                print("-" * 60)
            else:
                print("No past conversations found. Let's start a new one!")
                
        except Exception as e:
            print(f"Error retrieving history: {e}")

    async def process_user_input(self, user_input: str):
        """Process user input and get AI response"""
        try:
            print("Gigi is thinking...")
            response = await self.api.chat(message=user_input, user_id=self.user_id)
            
            if response['success']:
                self.conversation_count += 1
                return response['response']
            else:
                return f"Sorry, an error occurred: {response.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"A technical issue occurred: {e}"

    async def show_help(self):
        """Display help information"""
        print("\n" + "=" * 50)
        print("HELP - Available Commands")
        print("=" * 50)
        print("Just type a message to chat with me.")
        print("'history' - View your past conversation summaries.")
        print("'help' - Show this help message.")
        print("'exit' - Quit the application.")
        print("=" * 50)

    async def run(self):
        """Main terminal interaction loop"""
        self.print_welcome()

        while True:
            try:
                user_input = input(f"\n[{self.user_id}]: ").strip()

                if user_input.lower() == 'exit':
                    print(f"\nGoodbye, {self.user_id}! Come back anytime.")
                    break
                elif user_input.lower() == 'history':
                    await self.get_conversation_history()
                    continue
                elif user_input.lower() in ['help', '?']:
                    await self.show_help()
                    continue
                elif not user_input:
                    continue

                response = await self.process_user_input(user_input)

                print("\n" + "=" * 50)
                print("Gigi:")
                print("=" * 50)
                print(response)
                print("=" * 50)

            except KeyboardInterrupt:
                print(f"\nGoodbye, {self.user_id}!")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                continue

def main():
    """Entry point for the terminal agent"""
    parser = argparse.ArgumentParser(description="Gigi - Your Personal Growth Coach")
    parser.add_argument("--user", type=str, required=True, help="Your unique username")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    agent = TerminalGigiAgent(username=args.user)
    
    try:
        asyncio.run(agent.run())
    except Exception as e:
        print(f"Error starting agent: {e}")

if __name__ == "__main__":
    main()
