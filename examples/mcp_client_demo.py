"""MCP 客户端演示：验证 Server 握手 + 工具列表 + 调用 click_element。

运行：python examples/mcp_client_demo.py
（会拉起本项目的 MCP Server 子进程，通过 stdio 通信）
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "screen_use.mcp_server"],
        cwd=PROJECT_ROOT,
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"已连接，共 {len(tools.tools)} 个工具：")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description.splitlines()[0]}")

            print("\n调用 list_ui_elements ...")
            result = await session.call_tool("list_ui_elements", {})
            print(result.content[0].text[:500])


if __name__ == "__main__":
    asyncio.run(main())
