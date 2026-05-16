#include "NNE.h"
// (기타 헤더 생략)

// ... (모델 준비 및 데이터 평탄화 과정은 어제 코드와 동일) ...

// 1. 시간 측정 시작 (스톱워치 ON)
// FPlatformTime::Seconds()는 현재 엔진의 구동 시간을 초 단위(double)로 반환합니다.
double StartTime = FPlatformTime::Seconds();

// 2. AI 추론 실행 (CPU 또는 GPU 일꾼이 열심히 계산하는 구간)
// 여기서 코드가 잠시 멈추고 연산을 수행합니다.
if (Model->RunSync(InputBindings, OutputBindings) == 0) 
{
    // 3. 시간 측정 종료 (스톱워치 OFF)
    double EndTime = FPlatformTime::Seconds();
    
    // 4. 걸린 시간 계산 (초 단위를 밀리초(ms) 단위로 변환하기 위해 1000을 곱함)
    double ElapsedTimeMs = (EndTime - StartTime) * 1000.0;

    UE_LOG(LogTemp, Log, TEXT("AI 추론 성공! 걸린 시간: %f ms"), ElapsedTimeMs);
}
else
{
    UE_LOG(LogTemp, Error, TEXT("추론 실패..."));
}