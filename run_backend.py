import sys
import asyncio

# Set SelectorEventLoop on Windows to avoid WinError 10014 proactor accept bugs
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("Set WindowsSelectorEventLoopPolicy successfully.")

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
