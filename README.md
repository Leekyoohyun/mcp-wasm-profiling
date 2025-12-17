# EdgeAgent WASM Profiling

MCP Tool의 WASM 환경 실행시간을 세분화하여 측정하는 프로파일링 도구입니다.

## 목적

CCGrid-2026 논문을 위한 MCP Tool 성능 측정:
- **Cold Start vs Warm Start** 분리 측정
- **실행시간 분해**: T_io, T_serialize, T_compute
- **노드별 비교**: device-rpi, edge-nuc, edge-orin, cloud

## 측정 항목

```
T_exec = T_cold + T_io + T_serialize + T_compute

Where:
- T_cold: WASM 모듈 로드 + 초기화 (warm이면 ≈ 0)
- T_io: 파일/네트워크 I/O 시간
- T_serialize: JSON-RPC 파싱 + 응답 직렬화
- T_compute: 순수 연산 시간
```

## 디렉토리 구조

```
EdgeAgent-wasm-profiling/
├── README.md                    # 이 파일
├── docs/
│   └── methodology.md           # 측정 방법론 상세
├── scripts/
│   ├── measure_tools.py         # 메인 측정 스크립트
│   ├── analyze_results.py       # 결과 분석
│   └── setup_test_data.sh       # 테스트 데이터 준비
├── test_data/                   # 테스트용 파일들
│   ├── files/                   # 텍스트/JSON 파일
│   └── images/                  # 이미지 파일
├── results/                     # 측정 결과 (노드별)
│   ├── device-rpi/
│   ├── edge-nuc/
│   ├── edge-orin/
│   └── cloud/
└── wasm_modules/                # WASM 바이너리 (빌드 후 복사)
```

## 사전 요구사항

### 모든 노드에서 필요
- Python 3.8+
- wasmtime (`curl https://wasmtime.dev/install.sh -sSf | bash`)

### WASM 모듈 빌드 (개발 머신에서)
```bash
cd ~/EdgeAgent/wasm_mcp
cargo build --target wasm32-wasip2 --release
```

## 사용법

### 1. 저장소 클론 (각 노드에서)
```bash
git clone <repo_url> ~/EdgeAgent-wasm-profiling
cd ~/EdgeAgent-wasm-profiling
```

### 2. 테스트 데이터 준비
```bash
./scripts/setup_test_data.sh
```

### 3. WASM 모듈 복사
```bash
# 빌드된 WASM 모듈을 wasm_modules/ 디렉토리에 복사
cp ~/EdgeAgent/wasm_mcp/target/wasm32-wasip2/release/*.wasm ./wasm_modules/
```

### 4. 측정 실행
```bash
# 모든 tool 측정
python3 scripts/measure_tools.py

# 특정 tool만 측정
python3 scripts/measure_tools.py --tool read_file

# Cold start만 측정
python3 scripts/measure_tools.py --mode cold

# Warm start만 측정
python3 scripts/measure_tools.py --mode warm
```

### 5. 결과 분석
```bash
python3 scripts/analyze_results.py
```

## 측정 방식

### Cold Start 측정
매 실행마다 새로운 wasmtime 프로세스 생성:
```bash
wasmtime run --dir=/tmp tool.wasm < request.json
```

### Warm Start 측정 (연속 실행)
동일 프로세스에서 여러 요청 처리 (HTTP 모드):
```bash
# HTTP 서버 모드로 실행
wasmtime run --dir=/tmp tool-http.wasm &
curl -X POST http://localhost:8080 -d @request.json
```

### 시간 분해
1. **외부 측정**: 전체 실행 시간 (wall clock)
2. **내부 측정**: Tool 내부에서 I/O, serialize, compute 분리
   - 응답에 `_timing` 필드 포함

## 출력 형식

### 측정 결과 JSON
```json
{
  "tool_name": "read_file",
  "node": "edge-nuc",
  "mode": "cold",
  "input_size": 52428800,
  "output_size": 52521119,
  "runs": 5,
  "timing": {
    "total_ms": 14289.5,
    "cold_start_ms": 50.2,
    "io_ms": 14100.0,
    "serialize_ms": 89.3,
    "compute_ms": 50.0
  },
  "measurements": [14312, 14256, 14289, 14301, 14290]
}
```

## 참고

- Lumos 논문: "Performance Characterization of WebAssembly as a Serverless Runtime"
- MCP (Model Context Protocol): https://modelcontextprotocol.io/

## 관련 파일

- WASM MCP 서버: `~/EdgeAgent/wasm_mcp/`
- 기존 측정 결과: `~/final_result_for_scheduling/node-exectime-inputsize/`
