// 모듈 헤더
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"

void RunAIInference(TObjectPtr<UNNEModelData> MyModelData)
{
    // ---------------------------------------------------------
    // 1. 모델 준비
    // ---------------------------------------------------------
    FString RuntimeName = TEXT("NNERuntimeORTCpu"); 
    TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(RuntimeName);

    if (!Runtime.IsValid()) return; // 런타임을 못 찾으면 종료

    TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModel(MyModelData);
    if (!Model.IsValid()) return; // 모델 생성 실패 시 종료


    // ---------------------------------------------------------
    // 2. 입력 데이터 평탄화 (제일 중요!)
    // ---------------------------------------------------------
    // 언리얼 C++의 TArray는 다차원 구조를 이해하지 못함 
    // 따라서 데이터를 1차원으로 펴서 밀어넣어야 함
    
    TArray<float> InputData;
    
    // 예: [3, 224, 224] 크기의 3차원 이미지를 flatten()하여 
    // [150528] 길이의 1차원으로 일렬로 정렬
    // 파이토치에서 view(-1)로 한 줄로 정렬했던 것과 완전히 같은 원리
    int32 TotalInputSize = 3 * 224 * 224; 
    InputData.Init(0.5f, TotalInputSize); // 임시 가짜(Dummy) 데이터로 채움

    // 엔진이 요구하는 메모리 바인딩 형태로 포장
    TArray<UE::NNE::FTensorBindingCPU> InputBindings;
    InputBindings.Add({InputData.GetData(), (uint32)(InputData.Num() * sizeof(float))});


    // ---------------------------------------------------------
    // 3. 출력 데이터 받을 그릇 준비 및 추론 실행
    // ---------------------------------------------------------
    TArray<float> OutputData;
    // 1000개의 분류값을 뱉어내는 모델이라고 가정
    OutputData.SetNumUninitialized(1000); 

    TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
    OutputBindings.Add({OutputData.GetData(), (uint32)(OutputData.Num() * sizeof(float))});

    // RunSync: 동기식으로 추론을 멈춰서 실행 (결과가 나올 때까지 대기)
    if (Model->RunSync(InputBindings, OutputBindings) == 0) 
    {
        // ---------------------------------------------------------
        // 4. 결과 출력 및 해석
        // ---------------------------------------------------------
        int32 BestIndex = -1;
        float BestScore = -1.0f;

        // 결과 배열을 순회하며 가장 높은 점수를 찾는다
        for (int32 i = 0; i < OutputData.Num(); ++i)
        {
            if (OutputData[i] > BestScore)
            {
                BestScore = OutputData[i];
                BestIndex = i; // topk: 랭킹 1위 뽑기, 가장 확률이 높은 인덱스(ID)와 점수 획득 
            }
        }

        // 제각각인 점수를 확률값(0~1)으로 변환하려면, 이 결과에 Softmax 연산을 추가로 적용
        // (참고: 모델 내부(ONNX) 마지막 단에 Softmax가 포함되어 있다면 생략 가능)

        UE_LOG(LogTemp, Log, TEXT("AI 추론 성공! 랭킹 1위 인덱스: %d, 확률(점수): %f"), BestIndex, BestScore);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("추론 실패... 입력 크기나 모델 형식을 확인하세요."));
    }
}