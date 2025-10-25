#!/bin/bash

# MCP Server 테스트 스크립트

cd /home/ppak/pipe-inspector-electron

echo "🧪 Testing MCP Server..."
echo ""

# MCP 서버 테스트 (간단한 도구 호출)
cat << 'EOF' | conda run -n mcp-server python3 -c "
import sys
import json
import asyncio
from mcp_client import MCPClient

async def test():
    client = MCPClient('./mcp-server/server.py')

    try:
        print('📡 Connecting to MCP Server...')
        await client.connect()
        print('✅ Connected!\n')

        print('🔧 Listing available tools:')
        tools = await client.list_tools()
        for tool in tools:
            print(f'  - {tool.name}: {tool.description}')
        print()

        print('📊 Testing list_projects tool:')
        result = await client.call_tool('list_projects', {})
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()

asyncio.run(test())
"
EOF

echo ""
echo "✅ Test complete"
