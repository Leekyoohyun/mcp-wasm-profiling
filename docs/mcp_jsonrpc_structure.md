# MCP JSON-RPC Protocol Structure

MCP(Model Context Protocol)의 JSON-RPC 2.0 직렬화 구조 분석 문서입니다.

## 1. 전체 통신 흐름

```
┌─────────────┐                              ┌─────────────┐
│   Client    │                              │  MCP Server │
│ (LLM Agent) │                              │ (WASM Tool) │
└──────┬──────┘                              └──────┬──────┘
       │                                            │
       │  ① initialize (handshake)                  │
       │ ─────────────────────────────────────────► │
       │                                            │
       │  ② initialize response                     │
       │ ◄───────────────────────────────────────── │
       │                                            │
       │  ③ notifications/initialized               │
       │ ─────────────────────────────────────────► │
       │                                            │
       │  ④ tools/list (도구 목록 조회)              │
       │ ─────────────────────────────────────────► │
       │                                            │
       │  ⑤ tools/list response                     │
       │ ◄───────────────────────────────────────── │
       │                                            │
       │  ⑥ tools/call (실제 도구 호출)              │
       │ ─────────────────────────────────────────► │
       │                                            │
       │  ⑦ tools/call response                     │
       │ ◄───────────────────────────────────────── │
       │                                            │
```

## 2. 각 메시지 구조

### ① Initialize Request
```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "id": 1,
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {}
    },
    "clientInfo": {
      "name": "claude-code",
      "version": "1.0.0"
    }
  }
}
```

### ② Initialize Response
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": { "listChanged": false }
    },
    "serverInfo": {
      "name": "mcp-server-filesystem",
      "version": "0.1.0"
    }
  }
}
```

### ③ Initialized Notification (응답 없음)
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

### ④ Tools List Request
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 2
}
```

### ⑤ Tools List Response
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "read_file",
        "description": "Read the complete contents of a file...",
        "inputSchema": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to read"
            },
            "head": {
              "type": "integer",
              "description": "First N lines only"
            },
            "tail": {
              "type": "integer",
              "description": "Last N lines only"
            }
          },
          "required": ["path"]
        }
      },
      {
        "name": "write_file",
        "description": "Write content to a file...",
        "inputSchema": { ... }
      }
      // ... 14개 도구
    ]
  }
}
```

### ⑥ Tools Call Request (핵심)
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 3,
  "params": {
    "name": "read_file",
    "arguments": {
      "path": "/tmp/test.txt"
    }
  }
}
```

### ⑦ Tools Call Response
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Hello, World!\nThis is file content..."
      }
    ],
    "isError": false
  }
}
```

## 3. Serialization 오버헤드 분석

```
┌────────────────────────────────────────────────────────────────┐
│                    T_serialize 구성 요소                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Request (Input):                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Raw JSON bytes ──► serde_json::from_str() ──► Rust Struct│  │
│  │                                                          │  │
│  │ {"jsonrpc":"2.0","method":"tools/call",...}              │  │
│  │           │                                              │  │
│  │           ▼                                              │  │
│  │ struct JsonRpcRequest {                                  │  │
│  │     jsonrpc: String,      // "2.0"                       │  │
│  │     method: String,       // "tools/call"                │  │
│  │     id: Option<i64>,      // 3                           │  │
│  │     params: ToolCallParams,                              │  │
│  │ }                                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         T_parse                                │
│                                                                │
│  Response (Output):                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Rust Struct ──► serde_json::to_string() ──► JSON bytes   │  │
│  │                                                          │  │
│  │ struct ToolResult {                                      │  │
│  │     content: Vec<Content>,   // 실제 결과 데이터           │  │
│  │     is_error: bool,                                      │  │
│  │ }                                                        │  │
│  │           │                                              │  │
│  │           ▼                                              │  │
│  │ {"jsonrpc":"2.0","id":3,"result":{"content":[...]}}     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                       T_stringify                              │
│                                                                │
│  T_serialize = T_parse + T_stringify                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 4. 실제 WASM MCP 코드에서의 처리

```rust
// wasmmcp/src/protocol/jsonrpc.rs 에서 처리

// 1. Request 파싱
pub fn parse_request(input: &str) -> Result<JsonRpcRequest, Error> {
    serde_json::from_str(input)  // ◄── T_parse
}

// 2. Response 직렬화
pub fn serialize_response(response: &JsonRpcResponse) -> String {
    serde_json::to_string(response).unwrap()  // ◄── T_stringify
}
```

## 5. 입출력 크기별 Serialization 부하

| 시나리오 | Input JSON | Output JSON | 비고 |
|----------|------------|-------------|------|
| read_file (1KB) | ~100 bytes | ~1.4KB | Base overhead + content |
| read_file (1MB) | ~100 bytes | ~1.4MB | Output이 지배적 |
| read_file (50MB) | ~100 bytes | ~67MB | Base64면 +33% |
| read_media_file (1MB) | ~100 bytes | ~1.4MB | Base64 인코딩 |
| write_file (1MB) | ~1.4MB | ~50 bytes | Input이 지배적 |
| list_directory (1000) | ~80 bytes | ~50KB | 파일 개수 비례 |

## 6. Base64 인코딩 오버헤드 (read_media_file)

```
┌─────────────────────────────────────────────────────────────┐
│                  read_media_file 처리                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Binary File (1MB)                                          │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                        │
│  │ fs::read()      │◄── T_io (바이너리 읽기)                  │
│  │ Vec<u8>         │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ base64_encode() │◄── T_compute (인코딩)                   │
│  │ String (1.33MB) │    크기 33% 증가                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ JSON wrap       │◄── T_serialize                         │
│  └─────────────────┘                                        │
│                                                             │
│  Output: {"result":{"content":[{"type":"text",              │
│           "text":"data:image/png;base64,iVBOR..."}]}}       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 7. T_serialize 측정 포인트

```rust
// 측정 가능한 위치들
fn handle_tools_call(request: &str) -> String {
    // ─────────────────────────────────────────
    // T_parse 시작
    let start_parse = Instant::now();
    let req: JsonRpcRequest = serde_json::from_str(request)?;
    let params: ToolCallParams = serde_json::from_value(req.params)?;
    let t_parse = start_parse.elapsed();
    // T_parse 종료
    // ─────────────────────────────────────────

    // T_io + T_compute (도구 실행)
    let result = execute_tool(&params.name, &params.arguments)?;

    // ─────────────────────────────────────────
    // T_stringify 시작
    let start_stringify = Instant::now();
    let response = JsonRpcResponse {
        jsonrpc: "2.0",
        id: req.id,
        result: Some(result),
    };
    let output = serde_json::to_string(&response)?;
    let t_stringify = start_stringify.elapsed();
    // T_stringify 종료
    // ─────────────────────────────────────────

    output
}
```

## 8. JSON-RPC 오버헤드 특성 요약

| 항목 | 특성 | 영향 |
|------|------|------|
| **Protocol overhead** | 고정 ~100-200 bytes | 작은 응답에서 상대적으로 큼 |
| **serde_json 파싱** | O(n) | 입력 크기에 비례 |
| **serde_json 직렬화** | O(n) | 출력 크기에 비례 |
| **UTF-8 validation** | 텍스트 데이터 검증 | 추가 오버헤드 |
| **Base64 encoding** | 바이너리 → 텍스트 | 크기 33% 증가 |
| **Escape characters** | 특수문자 이스케이프 | JSON 안전성 |
