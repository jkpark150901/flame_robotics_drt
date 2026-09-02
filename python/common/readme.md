# Common Python Modules

## ZPipe Module (`zpipe.py`)

`zpipe.py`는 ZMQ(ZeroMQ)를 기반으로 한 통신 파이프라인 모듈입니다. Publish/Subscribe, Push/Pull, Router/Dealer 등의 다양한 패턴을 추상화하여 쉽게 사용할 수 있도록 제공합니다. Singleton 패턴으로 구현되어 있어 글로벌하게 파이프라인을 관리할 수 있습니다.

### 주요 기능
- **Singleton ZPipe Manager**: 전체 파이프라인과 Context를 관리합니다.
- **AsyncZSocket**: 비동기 소켓 래퍼로, 패턴에 따른 소켓 생성 및 통신을 담당합니다.
- **지원 패턴**: 'publish', 'subscribe', 'push', 'pull', 'dealer', 'router', 'server_pair', 'client_pair'

### 사용 예시

#### 1. 초기화 (Initialization)

사용하기 전에 반드시 파이프라인을 생성해야 합니다.

```python
from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe

# 파이프라인 생성 (IO thread 수 지정 가능, 기본 1)
pipeline = zpipe_create_pipe(io_threads=1)

# ... 애플리케이션 로직 ...

# 종료 시 파이프라인 파괴
zpipe_destroy_pipe()
```

#### 2. Publisher 예시

데이터를 발행(Publish)하는 예제입니다.

```python
import time
from common.zpipe import AsyncZSocket, zpipe_create_pipe

# 1. 파이프라인 생성
pipeline = zpipe_create_pipe()

# 2. Publisher 소켓 생성
pub_socket = AsyncZSocket("my_publisher", "publish")
pub_socket.create(pipeline)

# 3. 바인딩 (Bind) - Publisher는 일반적으로 Server 역할 (Bind)
# transport: tcp, ipc, inproc 등
pub_socket.join(transport="tcp", address="*", port=5555)

# 4. 데이터 전송
while True:
    try:
        topic = "sensor_data"
        message = "Hello ZMQ"
        
        # dispatch는 list 형태의 데이터를 받아서 multipart 메시지로 전송합니다.
        # 첫 번째 요소는 보통 토픽으로 사용됩니다.
        pub_socket.dispatch([topic, message])
        print(f"Sent: {topic} - {message}")
        
        time.sleep(1)
    except KeyboardInterrupt:
        break

# 5. 소켓 정리
pub_socket.close()
```

#### 3. Subscriber 예시

데이터를 수신(Subscribe)하는 예제입니다.

```python
import time
from common.zpipe import AsyncZSocket, zpipe_create_pipe

# 콜백 함수 정의
def on_message(data_parts):
    # data_parts는 bytes 리스트입니다.
    topic = data_parts[0].decode('utf-8')
    message = data_parts[1].decode('utf-8')
    print(f"Received: Topic={topic}, Message={message}")

# 1. 파이프라인 생성
pipeline = zpipe_create_pipe()

# 2. Subscriber 소켓 생성
sub_socket = AsyncZSocket("my_subscriber", "subscribe")
sub_socket.create(pipeline)

# 3. 연결 (Connect) - Subscriber는 일반적으로 Client 역할 (Connect)
sub_socket.join(transport="tcp", address="localhost", port=5555)

# 4. 콜백 설정 및 구독 (Subscribe)
sub_socket.set_message_callback(on_message)
sub_socket.subscribe("sensor_data")  # 특정 토픽 구독, 전체는 ""

# 5. 메인 루프 유지 (수신은 별도 스레드에서 동작함)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

# 6. 소켓 정리
sub_socket.close()
```
