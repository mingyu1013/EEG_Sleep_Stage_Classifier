def generate_feedback(preds):
    total = len(preds)
    feedback = []
    if total == 0:
        return ["예측 결과가 없습니다."]

    counts = {i: preds.count(i) for i in range(5)}
    ratios = {i: counts[i] / total for i in range(5)}

    def bullet(text):
        return "• " + text

    if ratios[3] < 0.15:
        feedback.append("⚠️ 깊은 수면(N3) 비율이 부족")
        feedback.extend([
            bullet("깊은 수면은 신체 회복과 면역력 강화에 중요한 역할을 합니다."),
            bullet("예측된 수면 결과에서 깊은 수면(N3)이 적게 나타났습니다."),
            bullet("자주 피로하고 아침에 개운하지 않다면 부족 가능성이 있습니다."),
            bullet("스마트폰 사용 줄이기, 방 온도 낮추기, 일정한 취침 시간을 추천합니다.")
        ])

    if ratios[4] == 0:
        feedback.append("⚠️ REM 수면이 감지되지 않음")
        feedback.extend([
            bullet("REM 수면은 꿈과 감정 정리를 담당하는 중요한 단계입니다."),
            bullet("스트레스, 불규칙한 수면, 카페인 섭취가 영향을 줄 수 있습니다."),
            bullet("REM 수면 부족은 기억력, 감정 기복, 집중력 저하와 관련 있습니다."),
            bullet("낮잠 줄이기, 동일한 시간에 잠들기 등이 회복에 도움이 됩니다.")
        ])

    if ratios[0] > 0.3:
        feedback.append("⚠️ 깨어 있는 시간 과다")
        feedback.extend([
            bullet("예측 결과에서 깨어 있는 비율이 높게 나타났습니다."),
            bullet("자는 도중 자주 깨거나 입면까지 시간이 오래 걸리는 경우일 수 있습니다."),
            bullet("취침 전 밝은 조명 줄이기, 스마트폰 사용 줄이기, 긴장 완화 습관이 도움이 됩니다.")
        ])

    if not feedback:
        feedback.append("✅ 수면 패턴이 안정적입니다.")
        feedback.extend([
            bullet("깊은 수면과 REM 수면 비율이 고르게 나타났습니다."),
            bullet("현재의 수면 습관을 잘 유지해보세요!")
        ])

    return feedback
