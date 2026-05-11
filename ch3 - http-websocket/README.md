# CH3 공부노트

> 💡Unreal NNE
> 
1. NNE란 무엇인가?
- NNE는 언리얼 엔진 5에 내장된 AI전용 통역기
- Pytorch: 블렌더, 마야(뭔가를 만드는 툴) / ONNX: FBX(포맷) / NNE: 언리얼 엔진의 스태틱 메시 렌더링(FBX 파일을 씬에 띄우고 실제로 움직이게 해주는 시스템)
- NNE는 무거운 Python 환경 없이도, 외부 서버의 도움 없이도 언리얼 엔진 스스로 ONNX 모델을 읽고 실행하게 해주는 시스템
2. NNE 아키텍처의 3대 핵심 구조
- 모델[설계도]
    - onnx 파일
    - AI가 어떻게 생각해야 하는지 적혀있는 설계도이며, NNE는 이 설계도를 언리얼 에디터의 에셋(Asset)으로 변환하여 저장
- 런타임[작업 영역 cpu vs gpu]
    - 설계도를 보고 실제로 계산(추론)을 수행하는 엔진 내부의 일꾼
    - 학습 기능이 제거된 추론 전용 구조로 되어 있어 속도가 매우 빠름
    - NNE의 가장 큰 장점은 RHI(DirectX/Vulkan)처럼 하드웨어별 최적화(DirectML 등)를 엔진이 알아서 수행한다는 것. 즉, 개발자가 복잡하게 세팅하지 않아도 CPU로 돌릴지, GPU로 가속해서 돌릴지 NNE가 알아서 교통정리를 해줌
- 텐서 인터페이스[데이터 포장 공간]
    - AI 모델에게 질문을 던지고(입력), 답변을 받는(출력) 곳
    - 주의점: 언리얼 C++의 TArray는 AI가 쓰는 다차원 구조를 이해하지 못함
    - 따라서 AI에게 데이터를 줄 때는 박스(다차원)를 분해해서 데이터를 1차원으로 일렬로 쫙 펴서(flatten()) 밀어 넣어야 함
---

> 💡NNE 개념 쉽게 정리
> 
| NNE 개념 | 중학생 비유 | 언리얼 엔지니어의 해석 |
| :--- | :--- | :--- |
| **ONNX Import** | USB에 담긴 게임 설치하기 | 에디터에서 ONNX 파일을 끌어다 놓아 UAsset으로 굽기 |
| **NNE Runtime** | 게임을 CPU로 돌릴지 그래픽카드로 돌릴지 런처에서 고르기 | `UNNEModelData`를 기반으로 CPU(Ort) 또는 GPU(DirectML) 추론 인터페이스 생성 |
| **Data Flatten** | 3단 도시락통을 해체해서 한 줄로 늘어놓기 | `TArray`의 1차원 특성에 맞춰 텐서를 `view(-1)` 또는 `flatten()`으로 평탄화하여 메모리 복사 |
---

> 💡NNE 활성화를 위한 언리얼 설정
> 
1. 에디터 설정: NNE 플러그인 활성화 (엔진에 통역사 채용하기)
- 언리얼 엔진은 기본적으로 무거워지는 것을 막기 위해 NNE 기능이 꺼져 있다
- 상단 메뉴 Edit(편집) > Plugins(플러그인) > NNE" (Neural Network Engine)를 검색해서 체크
- 주의점: NNE 관련 플러그인은 아직 '베타(Beta)'나 '실험단계(Experimental)'일 수 있다. 경고창이 뜨더라도 과감하게 Yes를 누르고 엔진을 재시작(Restart)
2. C++ 설정: Build.cs 모듈 추가 (코드에 통역사 초대하기)
- 에디터에서 켰다고 끝이 아니다. 우리가 작성할 C++ 코드가 NNE 기능(클래스나 함수)을 가져다 쓰려면, 프로젝트의 건축 설계도 격인 Build.cs 파일에 나 NNE 쓸 거야! 라고 명시해야 함(HTTP 통신 때도 똑같이 적용되는 중요한 개념)
- (프로젝트명).Build.cs 파일 -> PublicDependencyModuleNames 목록에 "NNE"를 추가
'''
C#
// 예시
PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "NNE" });
주의점: 이걸 빼먹으면 C++ 코드 상단에 #include "NNE.h"를 적는 순간 지옥의 컴파일 에러(LNK2019 - 확인할 수 없는 외부 참조)를 맛보게 됩니다.
'''
3. 에셋 호환성: ONNX 버전 맞추기 (통역사가 읽을 수 있는 책갈피 쓰기)
- 아무리 세팅을 잘해도 가져오는 ONNX 파일의 버전이 언리얼과 안 맞으면 엔진이 파일을 사용 불가
- 2주 차에 PyTorch에서 ONNX로 모델을 뽑아낼(Export) 때 썼던 opset_version=14 파라미터를 사용하여 FBX 버전 맞추듯 언리얼 호환성(NNE)을 맞춰야 함
- 현재 언리얼 NNE가 완벽하게 지원하는 ONNX Opset 버전을 확인하고 그에 맞춰서 내보내는 것이 중요
---

> 💡언리얼 버전이 지원하는 Opset 버전 확인
> 
1. 언리얼 엔진 공식 문서 (UE Documentation)
- 에픽게임즈 개발자 포털(dev.epicgames.com)에서 Neural Network Engine 또는 NNE를 검색
- 문서 내의 Supported Models (지원되는 모델) 또는 ONNX Runtime 섹션을 보면, 현재 엔진 버전(예: UE 5.3, UE 5.4 등)에서 테스트를 마친 권장 Opset 버전이 명시
2. 엔진 내장 ONNX Runtime 버전 역추적 (고급 엔지니어 방식)
- 엔진 설치 폴더의 플러그인 소스 코드: (Engine/Plugins/Experimental/NNE/Source/ThirdParty/...)를 뜯어보면, 현재 언리얼이 내장하고 있는 ONNX Runtime의 버전(예: 1.15.1 등)을 확인 가능
- 해당 ONNX Runtime 버전을 구글에 검색하여, ONNX 공식 깃허브의 '버전 호환성 표(Compatibility Matrix)'를 대조해 보면 몇 번 Opset까지 지원하는지 정확히 알 수 있다
---

> 💡NNE 프로젝트 세팅 가이드
> 
1. Edit(편집) > Plugins(플러그인) 메뉴에서 "NNE"를 검색
- Neural Network Engine 플러그인을 체크(Enable)
- 플러그인이 'Experimental(실험단계)'이라서 경고창이 뜰 수 있지만, "Yes"를 누르고 엔진을 재시작(Restart)
2. C++ 설정: Build.cs 모듈 추가
- 프로젝트 폴더 내 Source/(프로젝트이름)/(프로젝트이름).Build.cs에서 PublicDependencyModuleNames 목록에 "NNE"를 추가
'''
PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "NNE" });
'''
3. ONNX 호환성 확인: Opset 버전
- PyTorch에서 모델을 내보낼 때 opset_version=14를 사용했는지 확인
- NNE는 최신 FBX 버전을 맞춰야 언리얼이 읽을 수 있는 것처럼, 특정 Opset 버전을 기준으로 작동하기 때문
4. NNE 런타임 환경 이해
- CPU 런타임 (NNEModelRuntimeORT): 호환성이 가장 좋고 안정적
- GPU 런타임 (NNEModelRuntimeDML): 그래픽 카드를 사용하여 속도가 매우 빠르지만, 특정 하드웨어 설정이 필요
---

> 💡NNE Runtime 설정 가이드
> 
- 언리얼 엔진 내에서 CPU와 GPU 런타임 설정은 프로젝트 세팅 창에서 스위치 하나로 띡 켜고 끄는 방식이 아님
- 런타임은 어떤 플러그인을 활성화했는지와 코드(또는 블루프린트)에서 어떤 런타임을 호출하는지에 따라 개별적으로 결정
1. 플러그인(Plugins) 창에서 런타임 모듈 확인하기
- Edit > Plugins 창에서 "NNE"를 검색
- 기본 Neural Network Engine 외에 아래와 같은 런타임 플러그인들이 있는지 확인하고 체크(Enable)
- NNERuntimeORTCpu: CPU를 사용해 모델을 돌리기 위한 런타임 (안정성 최상)
- NNERuntimeORTDml: DirectML을 이용해 GPU 가속으로 모델을 돌리기 위한 런타임 (속도 최상)
2. 에셋(UAsset) 더블클릭해서 지원 여부 확인하기
- 생성된 에셋을 더블클릭
- 디테일(Details) 패널에 Supported Runtimes(지원되는 런타임)라는 항목 확인
- 모델의 구조나 호환성에 따라 엔진이 "이 모델은 NNERuntimeORTCpu로 돌릴 수 있음", "NNERuntimeORTDml도 지원함" 하고 목록을 보여준다. 여기서 내가 원하는 런타임이 목록에 뜨는지 눈으로 확인하는 것이 첫 번째 디버깅
3. C++ 코드(또는 블루프린트)에서 직접 배정
- C++ 코드나 블루프린트에서 모델을 '생성(Create Model Instance)'할 때, "나는 이 모델을 CPU 런타임(NNERuntimeORTCpu) 인터페이스로 열겠다!"라고 엔진에게 명시적으로 이름을 문자열(String)로 넘겨주게 됨
- 즉, 게임 내에서 A 모델은 무거우니 GPU로 돌리고, B 모델은 가벼우니 CPU로 돌리는 식으로 엔지니어가 코드 단에서 교통정리 가능
4. NNE 런타임 환경 이해
- CPU 런타임 (NNEModelRuntimeORT): 호환성이 가장 좋고 안정적
- GPU 런타임 (NNEModelRuntimeDML): 그래픽 카드를 사용하여 속도가 매우 빠르지만, 특정 하드웨어 설정이 필요
---