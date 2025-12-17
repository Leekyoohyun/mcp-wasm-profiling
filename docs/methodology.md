# WASM MCP Profiling Methodology

CCGrid-2026 논문을 위한 MCP Tool 실행시간 측정 방법론 문서입니다.

## 1. 개요

### 1.1 연구 목표

Edge-Cloud 환경에서 MCP(Model Context Protocol) Tool의 스케줄링 최적화를 위해,
각 노드에서의 실행시간을 정밀하게 측정하고 분해합니다.

### 1.2 측정 대상

```
T_exec = T_cold + T_io + T_serialize + T_compute
```

| 구성요소 | 설명 | 측정 방법 |
|----------|------|-----------|
| T_cold | WASM 모듈 로드 + 초기화 시간 | Cold vs Warm start 차이 |
| T_io | 파일/네트워크 I/O 시간 | 내부 타이밍 계측 |
| T_serialize | JSON-RPC 직렬화/역직렬화 | 내부 타이밍 계측 |
| T_compute | 순수 연산 시간 | T_exec - (T_cold + T_io + T_serialize) |

## 2. 측정 환경

### 2.1 대상 노드

| 노드명 | 아키텍처 | 특징 |
|--------|----------|------|
| device-rpi | ARM (Raspberry Pi) | IoT 디바이스 |
| edge-nuc | x86_64 (Intel NUC) | Edge 서버 |
| edge-orin | ARM (Jetson Orin) | AI Edge 서버 |
| cloud | x86_64 (AWS/GCP) | 클라우드 서버 |

### 2.2 런타임 환경

```
WASM Runtime: wasmtime (latest)
Target: wasm32-wasip2
Transport: Stdio (cold) / HTTP (warm)
```

## 3. 측정 방식

### 3.1 Cold Start 측정

매 실행마다 새로운 wasmtime 프로세스를 생성하여 전체 초기화 오버헤드를 포함합니다.

```
┌──────────────────────────────────────────────────────────────┐
│                    Cold Start 측정 흐름                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [Start Timer]                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                         │
│  │ wasmtime 프로세스 │◄── 프로세스 생성                         │
│  │    시작          │                                         │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ WASM 모듈 로드    │◄── T_cold의 주요 부분                    │
│  │ & 초기화         │                                         │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ MCP Initialize   │◄── JSON-RPC initialize                 │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ tools/call 실행   │◄── 실제 Tool 실행                       │
│  │ (I/O + Compute)  │                                        │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  [Stop Timer]                                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### 실행 명령어

```bash
echo '$JSON_RPC_REQUEST' | wasmtime run --dir=/tmp tool.wasm
```

### 3.2 Warm Start 측정

HTTP 모드에서 이미 실행 중인 서버에 요청을 보내 순수 실행 시간만 측정합니다.

```
┌──────────────────────────────────────────────────────────────┐
│                    Warm Start 측정 흐름                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [서버 사전 실행]                                              │
│  wasmtime serve --addr 127.0.0.1:8000 -S cli=y \            │
│    --dir=/tmp tool-http.wasm                                 │
│                                                              │
│  [Start Timer]                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                         │
│  │ HTTP POST 요청   │◄── 네트워크 지연 최소화 (localhost)        │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ tools/call 실행   │◄── T_cold ≈ 0                          │
│  │ (I/O + Compute)  │    (모듈 이미 로드됨)                     │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  [Stop Timer]                                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### 실행 명령어

```bash
# 서버 시작 (한 번)
wasmtime serve --addr 127.0.0.1:8000 -S cli=y --dir=/tmp tool-http.wasm &

# 측정 (여러 번)
curl -X POST http://localhost:8000 -d @request.json
```

### 3.3 Cold Start 시간 계산

```
T_cold = T_exec_cold - T_exec_warm
```

동일한 입력에 대해 Cold Start와 Warm Start 실행시간 차이로 계산합니다.

## 4. 시간 분해 (Time Decomposition)

### 4.1 Lumos 논문 기반 방법론

Lumos 논문의 방법론을 참고하여 실행시간을 분해합니다:

```
T_exec = T_cold + T_data_retrieval + T_serialization + T_compute
```

### 4.2 내부 계측 (Instrumentation)

Tool 코드 내부에서 각 단계의 시간을 측정하여 응답에 포함합니다.

```rust
// 예시: read_file tool 내부 계측
fn read_file(params: ReadFileParams) -> Result<Response, Error> {
    let start = Instant::now();

    // I/O 측정
    let io_start = Instant::now();
    let content = fs::read_to_string(&params.path)?;
    let io_ms = io_start.elapsed().as_secs_f64() * 1000.0;

    // Serialize 측정
    let serialize_start = Instant::now();
    let response = serde_json::to_string(&content)?;
    let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let compute_ms = total_ms - io_ms - serialize_ms;

    Ok(Response {
        result: response,
        _timing: Timing {
            io_ms,
            serialize_ms,
            compute_ms,
        },
    })
}
```

### 4.3 JSON-RPC 직렬화 오버헤드

MCP는 JSON-RPC 2.0 프로토콜을 사용합니다:

```
┌────────────────────────────────────────────────────────────┐
│                  JSON-RPC 처리 흐름                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Request:                                                  │
│  {"jsonrpc":"2.0","method":"tools/call","params":{...}}   │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────────────┐                                      │
│  │ JSON 파싱         │◄── T_serialize (input)              │
│  │ (serde_json)      │                                     │
│  └────────┬─────────┘                                      │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐                                      │
│  │ Tool 실행         │◄── T_io + T_compute                  │
│  └────────┬─────────┘                                      │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐                                      │
│  │ JSON 직렬화       │◄── T_serialize (output)              │
│  │ (serde_json)      │                                     │
│  └────────┬─────────┘                                      │
│           │                                                │
│           ▼                                                │
│  Response:                                                 │
│  {"jsonrpc":"2.0","result":{...},"id":1}                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 5. Tool별 특성

### 5.1 I/O 집약형 (I/O-bound)

| Tool | 주요 I/O 작업 |
|------|--------------|
| read_file | 파일 읽기 |
| read_text_file | 텍스트 파일 읽기 |
| read_media_file | 바이너리 파일 읽기 + Base64 인코딩 |
| write_file | 파일 쓰기 |
| read_multiple_files | 다중 파일 읽기 |

### 5.2 디렉토리 탐색형

| Tool | 주요 작업 |
|------|----------|
| list_directory | readdir syscall |
| list_directory_with_sizes | readdir + stat |
| directory_tree | 재귀적 readdir |
| search_files | 재귀적 readdir + 패턴 매칭 |

### 5.3 메타데이터형

| Tool | 주요 작업 |
|------|----------|
| get_file_info | stat syscall |

## 6. 측정 파라미터

### 6.1 입력 크기 (Input Size)

| 카테고리 | 크기 |
|----------|------|
| 소형 | 1KB, 10KB |
| 중형 | 100KB, 1MB |
| 대형 | 10MB, 50MB |

### 6.2 반복 횟수

| 타입 | 횟수 | 용도 |
|------|------|------|
| Warmup | 3회 | 캐시 준비 |
| Measurement | 10회 | 실제 측정 |

### 6.3 통계 처리

- **평균 (Mean)**: 대표값
- **표준편차 (Std)**: 변동성
- **최소/최대 (Min/Max)**: 범위

## 7. 결과 포맷

### 7.1 개별 측정 결과

```json
{
  "tool_name": "read_file",
  "node": "edge-nuc",
  "mode": "cold",
  "input_size": 1048576,
  "input_size_label": "1MB",
  "output_size": 1398101,
  "runs": 10,
  "timestamp": "2024-12-17T15:30:00",
  "timing": {
    "total_ms": 156.8,
    "cold_start_ms": 45.2,
    "io_ms": 98.5,
    "serialize_ms": 8.1,
    "compute_ms": 5.0
  },
  "timing_std": {
    "total_ms": 12.3,
    "cold_start_ms": 3.1,
    "io_ms": 8.7,
    "serialize_ms": 0.5,
    "compute_ms": 0.3
  },
  "measurements": [152.1, 158.3, 155.2, ...]
}
```

### 7.2 비교 분석 결과

```json
{
  "comparison": {
    "cold_vs_warm": {
      "read_file:1MB": {
        "cold_mean_ms": 156.8,
        "warm_mean_ms": 111.6,
        "speedup": 1.4,
        "cold_overhead_ms": 45.2
      }
    }
  },
  "decomposition": {
    "read_file:1MB": {
      "cold_start_pct": 28.8,
      "io_pct": 62.8,
      "serialize_pct": 5.2,
      "compute_pct": 3.2
    }
  }
}
```

## 8. 스케줄링 활용

### 8.1 Alpha (α) 계산

```
α = T_exec / (T_exec + T_comm_ref)

T_comm_ref = (input_size + output_size) / bandwidth + latency
```

- **α → 1**: 실행 지배적 (로컬 실행 선호)
- **α → 0**: 통신 지배적 (오프로딩 고려)

### 8.2 bound_type 분류

측정된 시간 분해를 기반으로 Tool 유형 분류:

```python
if io_pct > 60:
    bound_type = "io-bound"
elif compute_pct > 60:
    bound_type = "compute-bound"
else:
    bound_type = "mixed"
```

### 8.3 노드별 P_comp 정규화

CoreMark 점수를 기준으로 순수 연산 능력 정규화:

```
P_comp_normalized = T_compute * (CoreMark_ref / CoreMark_node)
```

## 9. 참고 문헌

1. **Lumos**: "Performance Characterization of WebAssembly as a Serverless Runtime"
   - 시간 분해 방법론 참고
   - Cold/Warm start 구분

2. **MCP Specification**: https://modelcontextprotocol.io/
   - JSON-RPC 2.0 프로토콜
   - Tool call 인터페이스

3. **Wasmtime**: https://wasmtime.dev/
   - WASM 런타임 특성
   - WASI 인터페이스

## 10. 부록: 측정 스크립트 사용법

### A. 전체 측정

```bash
# 테스트 데이터 준비
./scripts/setup_test_data.sh

# WASM 모듈 복사
cp ~/EdgeAgent/wasm_mcp/target/wasm32-wasip2/release/*.wasm ./wasm_modules/

# 측정 실행
python3 scripts/measure_tools.py

# 결과 분석
python3 scripts/analyze_results.py
```

### B. 특정 Tool 측정

```bash
# read_file만, 1MB 입력, Cold start
python3 scripts/measure_tools.py --tool read_file --size 1MB --mode cold

# Warm start (서버 먼저 실행)
wasmtime serve --addr 127.0.0.1:8000 -S cli=y --dir=/tmp \
  ./wasm_modules/mcp_server_filesystem_http.wasm &

python3 scripts/measure_tools.py --tool read_file --mode warm
```

### C. 다중 노드 측정

```bash
# 각 노드에서
git clone <repo> ~/EdgeAgent-wasm-profiling
cd ~/EdgeAgent-wasm-profiling
./scripts/setup_test_data.sh
python3 scripts/measure_tools.py

# 결과 수집 후 분석
python3 scripts/analyze_results.py --compare --decomposition
```
