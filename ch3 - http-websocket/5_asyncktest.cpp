#include "Async/Async.h"
// ... (기타 NNE 헤더 등 포함) ...

// ==============================================================================
// [방어력 3: 델리게이트] W6에서 심화로 다룰 내용입니다.
// 백그라운드에서 연산이 끝났을 때, 블루프린트 등 게임 로직으로 결과를 "방송(Broadcast)" 해줍니다.
// ==============================================================================
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnInferenceFinished, int32, BestIndex, float, BestScore);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class UMyAIInferenceComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    // 블루프린트의 이벤트 그래프에서 빨간색 노드로 꺼내 쓸 수 있는 이벤트
    UPROPERTY(BlueprintAssignable, Category = "AI")
    FOnInferenceFinished OnInferenceFinished;

    void RunInferenceAsync()
    {
        // ==============================================================================
        // [방어력 2: TWeakObjectPtr] 가비지 컬렉션(GC) 및 댕글링 포인터 대비
        // 비동기 작업(AI 연산)이 2~3초 걸린다고 가정했을 때, 그 사이 이 컴포넌트(혹은 몬스터)가 
        // 플레이어에게 죽어 파괴될 수 있습니다. 이를 안전하게 추적하기 위해 약한 참조를 만듭니다.
        // ==============================================================================
        TWeakObjectPtr<UMyAIInferenceComponent> WeakThis(this);

        // 1. 메인 스레드(Game Thread)를 멈추지 않게 하기 위해, 별도의 워커 스레드로 작업을 던집니다.
        AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [WeakThis]()
        {
            // 🚨 주의: 이 블록 안은 '워커 스레드'입니다. 
            // 절대 여기서 UI 텍스트를 바꾸거나 파티클을 스폰하는 등 언리얼 씬(Scene)을 건드리면 안 됩니다!
            
            // -------- [무거운 NNE AI 추론 연산 실행 (동기식 RunSync)] --------
            FPlatformProcess::Sleep(2.0f); // (예시) 연산에 2초가 걸렸다고 가정
            int32 ResultIndex = 1;
            float ResultScore = 0.98f;
            // ---------------------------------------------------------------


            // ==============================================================================
            // [방어력 1: 게임 스레드 복귀] Game Thread 규칙
            // AI 연산이 끝났으니, 결과값(ResultIndex, ResultScore)을 들고 안전한 메인 스레드로 돌아갑니다.
            // ==============================================================================
            AsyncTask(ENamedThreads::GameThread, [WeakThis, ResultIndex, ResultScore]()
            {
                // ✅ 이 블록부터는 다시 메인 게임 스레드입니다. 액터나 UI를 만져도 안전합니다.

                // 방어력 2의 진가: 2초의 연산 시간 동안 내가 속한 액터가 죽지 않고 살아있는지 확인합니다.
                if (WeakThis.IsValid())
                {
                    UE_LOG(LogTemp, Log, TEXT("AI 연산 완료! 메인 스레드 복귀 성공."));
                    
                    // 살아있다면, 방어력 3(델리게이트)을 발동하여 게임 로직에 결과를 넘깁니다.
                    // 이 코드가 실행되면 블루프린트에 연결된 다음 노드(예: 공격 애니메이션 재생)가 실행됩니다.
                    WeakThis->OnInferenceFinished.Broadcast(ResultIndex, ResultScore);
                }
                else
                {
                    // 액터가 이미 파괴되었다면, 결과를 깔끔하게 무시합니다. (치명적인 크래시 방어 성공!)
                    UE_LOG(LogTemp, Warning, TEXT("추론은 완료되었으나 액터가 이미 파괴되었습니다. 결과를 폐기합니다."));
                }
            });

        });
    }
};