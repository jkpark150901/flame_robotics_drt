# Viewervedo ZAPI Usage Guide

이 문서는 `viewervedo` 모듈에서 제공하는 ZAPI(ZeroMQ API) 수신 가능 함수들에 대한 명세와 사용 방법(Usage)을 설명합니다.

`viewervedo`의 ZAPI는 `ROUTER` 패턴으로 동작하며, 클라이언트(예: `simtool`)가 특정 이벤트를 요청하면 내부의 `_dispatch_message` 메커니즘을 통해 `zapi_` 접두어가 붙은 함수를 자동으로 찾아 실행합니다.

---

## 🚀 사용 가능한 ZAPI 핸들러 (Handler Functions)

클라이언트에서 `ZAPIBase.call(socket, "함수명", kwargs)` 형식으로 메시지를 전송하면, 뷰어의 `zapi_<함수명>` 함수가 실행됩니다.

### 1. `zapi_terminate`
뷰어 애플리케이션의 종료를 요청합니다.
- **인자 (kwargs)**
  - `payload` (선택): 전달할 메시지나 데이터 (현재는 사용하지 않음)
- **응답 (Reply)**
  - 없음 (뷰어 종료됨)
- **호출 예시 (Client)**
  ```python
  zapi.call(socket, "terminate", {})
  ```

### 2. `zapi_ping`
뷰어 애플리케이션의 연결 상태를 확인하는 Ping 테스트입니다.
- **인자 (kwargs)**
  - 없음 (내부적으로 ZMQ `_identity` 정보가 추가됨)
- **응답 (Reply)**
  - 함수명: `"pong"`
  - 데이터: `{}` (빈 딕셔너리)
- **호출 예시 (Client)**
  ```python
  zapi.call(socket, "ping", {})
  ```

### 3. `zapi_load_spool`
Spool 데이터(`.pcd`, `.ply`) 파일을 로드하여 화면에 시각화하도록 요청합니다. 백그라운드 스레드에서 시각화 메인 스레드로 렌더링 큐를 통해 명령이 푸시(push)됩니다.
- **인자 (kwargs)**
  - `path` (str, 필수): 로드할 스풀 데이터 파일의 경로 (절대/상대 경로)
- **응답 (Reply)**
  - 함수명: `"reply_load_spool"`
  - 데이터 (`status`에 성공 여부 반환)
    ```json
    {
      "path": "/전달된/파일/경로.pcd",
      "status": "success" // 또는 "failed"
    }
    ```
- **호출 예시 (Client)**
  ```python
  zapi.call(socket, "load_spool", {"path": "/Users/.../sample.pcd"})
  ```

---

## 🛠 `viewervedo` 내부 구현 예시 (새로운 API 추가 방법)

ZAPI 핸들러를 추가하고 싶을 경우, `viewervedo/zapi.py` 파일의 `ZAPI` 클래스에 다음과 같이 `zapi_` 접두어가 붙은 함수를 정의하기만 하면 됩니다.

```python
class ZAPI(ZAPIBase):
    # ... 기존 코드 ...

    def zapi_my_custom_action(self, kwargs=None):
        """새로운 기능: my_custom_action"""
        data = kwargs.get("data")
        self.__console.info(f"Custom action called with data: {data}")
```

이후 클라이언트에서는 `"my_custom_action"` 이라는 이름으로 `call` 하면 자동으로 매핑되어 실행됩니다.
