# SimTool ZAPI Usage Guide

이 문서는 `simtool` 모듈에서 제공하는 ZAPI(ZeroMQ API)에 대한 명세와 사용 방법(Usage)을 설명합니다.

`simtool`의 ZAPI는 뷰어(`viewervedo`)와 데이터를 주고받는 클라이언트(`DEALER`) 패턴으로 동작합니다. 따라서 직접적으로 `zapi_` 접두어가 붙은 수신 핸들러 함수를 가지는 대신, **PyQt6 Signal 기반으로 수신 메시지를 처리**하며, 뷰어로 전송하는 **호출(Request) 함수**들을 제공합니다.

---

## 🚀 ZAPI 송신용 함수 (Request Methods)

`simtool` 에서 `viewervedo` 로 명령을 내릴 때 사용하는 래퍼(Wrapper) 함수들입니다. `simtool/zapi.py` 내부의 `ZAPI` 클래스에 정의되어 UI/로직 코드에서 사용됩니다.

### 1. `_ZAPI_request_load_spool`
뷰어에게 특정 Spool 파일(`.pcd`, `.ply`)을 로드하여 화면에 렌더링하도록 요청합니다.
- **인자 (Arguments)**
  - `file_path` (str, 필수): 로드할 스풀 데이터 파일의 경로
- **내부 동작**
  뷰어 측 `load_spool` 핸들러로 ZMQ 메시지를 발송합니다.
- **호출 예시 (SimTool 내부)**
  ```python
  # (simtool/window.py 등에서 호출)
  if self.zapi:
      self.zapi._ZAPI_request_load_spool("/User/.../sample_spool.pcd")
  ```

> 💡 **추가 기능 구현 시**
> 뷰어(`viewervedo`)에 새로운 이벤트를 보내야 한다면 `simtool/zapi.py` 에 이와 비슷한 형식으로 `_ZAPI_request_<새로운기능이름>` 메서드를 추가하여 캡슐화하는 것이 좋습니다.

---

## 📥 ZAPI 수신 처리 구조 (Signal/Slot)

`simtool` 에서는 비동기로 수신된 데이터(Reply 메시지 등)를 PyQt6의 Signal & Slot 메커니즘을 통해 메인 스레드로 안전하게 넘겨 처리합니다.

### 이벤트 수신 메커니즘

1. `_on_message_received` 데이터 콜백이 수신된 데이터를 파싱합니다.
2. `signal_message_received(topic, msg_json)` 시그널을 `emit` 합니다.
3. `simtool/window.py` 의 **`_handle_message(self, topic, msg)`** 메서드(Slot)가 작동하여 이벤트를 처리합니다.

### 현재 처리되는 수신 메시지(Topic) 목록

#### 수신 Topic: `call` -> Command: `"reply_load_spool"`
뷰어에 특정 파일을 로드해달라고 전송한 뒤, 로드가 성공/실패했음을 알려주는 응답 메시지입니다.
- **JSON 메시지 예시**
  ```json
  {
    "command": "reply_load_spool",
    "path": "/전달된/파일/경로.pcd",
    "status": "success"
  }
  ```
- **동작**: 상태(`status`)가 `"success"`일 경우 팝업 메시지를 띄우거나, 에러 로그를 출력합니다.
