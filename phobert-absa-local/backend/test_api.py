import json
import logging
from pydantic import ValidationError
from fastapi import HTTPException
from backend.app import app, health, aspects, analyze, AnalyzeRequest, ASPECTS, LABELS

logging.basicConfig(level=logging.INFO)

def run_tests():
    print("=== 1. TEST GET /api/health (trước khi inference) ===")
    h1 = health()
    print("Health response:", h1)
    assert h1["status"] == "ok"
    assert h1["checkpoint_exists"] is True
    assert "device" in h1
    assert "model_loaded" in h1
    assert "model_ready" in h1

    print("\n=== 2. TEST GET /api/aspects ===")
    asp = aspects()
    print("Aspects response:", asp)
    assert asp["aspects"] == ASPECTS
    assert len(asp["aspects"]) == 10

    print("\n=== 3. TEST POST /api/analyze - Test Case 1 ===")
    req1 = AnalyzeRequest(
        comment="Camera chụp ảnh rất đẹp nhưng pin nhanh hết.",
        aspects=["CAMERA", "BATTERY"]
    )
    res1 = analyze(req1)
    print("Test Case 1 Response:")
    print(json.dumps(res1, indent=2, ensure_ascii=False))
    assert len(res1["results"]) == 2
    res_map_1 = {r["aspect"]: r for r in res1["results"]}
    assert res_map_1["CAMERA"]["sentiment"] == "Positive"
    assert res_map_1["BATTERY"]["sentiment"] == "Negative"
    assert res_map_1["CAMERA"]["confidence"] > 0.9
    assert res_map_1["BATTERY"]["confidence"] > 0.9

    print("\n=== 4. TEST POST /api/analyze - Test Case 2 ===")
    req2 = AnalyzeRequest(
        comment="Màn hình đẹp, hiệu năng mạnh nhưng máy hơi nóng.",
        aspects=["SCREEN", "PERFORMANCE"]
    )
    res2 = analyze(req2)
    print("Test Case 2 Response:")
    print(json.dumps(res2, indent=2, ensure_ascii=False))
    assert len(res2["results"]) == 2
    res_map_2 = {r["aspect"]: r for r in res2["results"]}
    assert res_map_2["SCREEN"]["sentiment"] == "Positive"
    assert res_map_2["PERFORMANCE"]["sentiment"] in ["Neutral", "Positive", "Negative"]

    print("\n=== 5. TEST POST /api/analyze - Test Case 3 ===")
    req3 = AnalyzeRequest(
        comment="Điện thoại sử dụng ổn, thiết kế bình thường.",
        aspects=["DESIGN"]
    )
    res3 = analyze(req3)
    print("Test Case 3 Response:")
    print(json.dumps(res3, indent=2, ensure_ascii=False))
    assert len(res3["results"]) == 1
    assert res3["results"][0]["aspect"] == "DESIGN"
    assert res3["results"][0]["sentiment"] == "Neutral"
    assert res3["results"][0]["confidence"] > 0.8

    print("\n=== 6. TEST Validation - Empty / Whitespace comment ===")
    # Trường hợp 1: comment rỗng theo pydantic Field(min_length=1)
    try:
        AnalyzeRequest(comment="", aspects=["CAMERA"])
        assert False, "Should have raised ValidationError for empty comment"
    except ValidationError as e:
        print("Pydantic caught empty comment successfully:", type(e).__name__)

    # Trường hợp 2: comment chỉ chứa dấu cách
    try:
        req_space = AnalyzeRequest(comment="   ", aspects=["CAMERA"])
        analyze(req_space)
        assert False, "Should have raised HTTPException for whitespace comment"
    except HTTPException as e:
        print("analyze() caught whitespace comment successfully:", e.status_code, e.detail)
        assert e.status_code == 422

    print("\n=== 7. TEST Validation - Invalid aspect ===")
    try:
        req_invalid = AnalyzeRequest(comment="Máy tốt", aspects=["INVALID_ASPECT"])
        analyze(req_invalid)
        assert False, "Should have raised HTTPException for invalid aspect"
    except HTTPException as e:
        print("analyze() caught invalid aspect successfully:", e.status_code, e.detail)
        assert e.status_code == 422

    print("\n=== 8. TEST Validation - Empty aspects list ===")
    try:
        AnalyzeRequest(comment="Máy tốt", aspects=[])
        assert False, "Should have raised ValidationError for empty aspects list"
    except ValidationError as e:
        print("Pydantic caught empty aspects list successfully:", type(e).__name__)

    print("\n=== 9. TEST Long comment & All 10 Aspects ===")
    long_comment = "Điện thoại này dùng rất mượt, camera nét, loa to, màn hình sáng đẹp, pin trâu cả ngày, giá cả hợp lý, chế độ bảo hành chu đáo, kiểu dáng hiện đại thanh lịch, dung lượng bộ nhớ thoải mái lưu trữ." * 5
    req_long = AnalyzeRequest(
        comment=long_comment,
        aspects=ASPECTS
    )
    res_long = analyze(req_long)
    print(f"Long comment analyzed successfully. Results count: {len(res_long['results'])}")
    assert len(res_long["results"]) == 10
    for r in res_long["results"]:
        assert r["aspect"] in ASPECTS
        assert r["sentiment"] in ["Negative", "Neutral", "Positive"]
        assert 0 <= r["confidence"] <= 1.0

    print("\n=== 10. TEST Deduplication of aspects ===")
    req_dup = AnalyzeRequest(comment="Máy tốt", aspects=["CAMERA", "CAMERA", "BATTERY"])
    res_dup = analyze(req_dup)
    assert len(res_dup["results"]) == 2
    assert [r["aspect"] for r in res_dup["results"]] == ["CAMERA", "BATTERY"]
    print("Deduplication passed:", [r["aspect"] for r in res_dup["results"]])

    print("\n=== 11. TEST GET /api/health (sau khi inference) ===")
    h2 = health()
    print("Health response after inference:", h2)
    assert h2["model_loaded"] is True
    assert h2["model_ready"] is True

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_tests()
