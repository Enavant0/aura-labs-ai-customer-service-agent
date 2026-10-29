"""本地评分与选题回归检查，不加载 BGE，不调用 DeepSeek，不打开 Chroma。"""

from contextlib import redirect_stdout, redirect_stderr
import csv
from io import StringIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_rules import parse_concept_groups, score_answer
from scripts import eval_rag


def make_case(action="ANSWER", required="", forbidden="继续使用", case_id="TC-001") -> dict:
    return {"case_id": case_id, "category": "评分单元测试", "user_question": "测试问题",
            "expected_action": action, "expected_sources": "test.md", "reference_answer": "测试参考答案",
            "must_include": required, "must_not_include": forbidden}


class ForbiddenContextTests(unittest.TestCase):
    def test_requested_four_examples(self):
        for answer, forbidden, expected in (
            ("不建议继续使用", "继续使用", True),
            ("建议继续使用", "继续使用", False),
            ("不能直接承诺补发", "直接承诺补发", True),
            ("可以直接承诺补发", "直接承诺补发", False),
        ):
            with self.subTest(answer=answer):
                score = score_answer(make_case(forbidden=forbidden), answer)
                self.assertEqual(score["must_not_include_pass"], expected)
                self.assertEqual(score["rule_pass"], expected)

    def test_refund_and_allergy_guarantees(self):
        for answer, forbidden, expected in (
            ("不能承诺全额退款", "承诺全额退款", True),
            ("可以承诺全额退款", "承诺全额退款", False),
            ("不能保证百分百不过敏", "百分百不过敏", True),
            ("保证百分百不过敏", "百分百不过敏", False),
            ("不能保证“百分百不过敏”", "百分百不过敏", True),
            ("不是医生级疗效", "医生级疗效", True),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(score_answer(make_case(forbidden=forbidden), answer)["rule_pass"], expected)

    def test_common_negations_and_polite_modifiers(self):
        for negation in ("不", "不能", "不要", "不得", "禁止", "不建议", "不支持", "不应", "无法", "不可"):
            with self.subTest(negation=negation):
                self.assertTrue(score_answer(make_case(), negation + "继续使用")["must_not_include_pass"])
        self.assertTrue(score_answer(make_case(), "我们不建议您再继续使用。")["must_not_include_pass"])

    def test_every_occurrence_is_checked(self):
        score = score_answer(make_case(), "不建议继续使用。但是可以继续使用。")
        self.assertFalse(score["rule_pass"])
        detail = score["rule_details"]["must_not_include"][0]
        self.assertEqual([item["exempted"] for item in detail["occurrences"]], [True, False])
        self.assertFalse(detail["exempted"])
        self.assertTrue(detail["violation"])

    def test_negation_does_not_cross_clauses_or_turns(self):
        for answer in ("不能保证效果。建议继续使用。", "不能保证效果，建议继续使用。",
                       "不能保证效果\n建议继续使用。", "不能保证效果但建议继续使用。"):
            with self.subTest(answer=answer):
                self.assertFalse(score_answer(make_case(), answer)["must_not_include_pass"])

    def test_unrelated_bu_and_double_negation_do_not_exempt(self):
        for answer in ("不舒服建议继续使用", "不妨继续使用", "不仅建议继续使用", "不是不建议继续使用", "不能不继续使用"):
            with self.subTest(answer=answer):
                self.assertFalse(score_answer(make_case(), answer)["must_not_include_pass"])

    def test_rule_details_include_hits_misses_and_exemptions(self):
        score = score_answer(make_case(required="停止使用;转人工", forbidden="继续使用;全额退款"),
                             "请停止使用，不建议继续使用。")
        details = score["rule_details"]
        self.assertEqual([item["concept_group"] for item in details["must_include"]], ["停止使用", "转人工"])
        self.assertEqual([item["matched"] for item in details["must_include"]], [True, False])
        self.assertEqual(details["must_include"][0]["matched_expressions"], ["停止使用"])
        self.assertTrue(details["must_not_include"][0]["detected"])
        self.assertTrue(details["must_not_include"][0]["exempted"])
        self.assertIn("不建议", details["must_not_include"][0]["occurrences"][0]["negation"])
        self.assertFalse(details["must_not_include"][1]["detected"])
        self.assertTrue(details["expected_action"]["passed"])


class ActionAndSelectionTests(unittest.TestCase):
    def test_refusal_variants(self):
        for answer in ("无法确认", "不能确认", "没有公开信息", "无法提供", "不能判断"):
            with self.subTest(answer=answer):
                self.assertTrue(score_answer(make_case("REFUSE", forbidden=""), answer)["action_pass"])

    def test_handoff_variants(self):
        for answer in ("转人工", "人工客服", "人工核实", "售后审核"):
            with self.subTest(answer=answer):
                self.assertTrue(score_answer(make_case("HANDOFF", forbidden=""), answer)["action_pass"])

    def test_tool_variants(self):
        for answer in ("查询业务系统", "查询订单系统", "查询物流系统", "查询售后系统", "转人工", "人工核实"):
            with self.subTest(answer=answer):
                self.assertTrue(score_answer(make_case("HANDOFF_OR_TOOL", forbidden=""), answer)["action_pass"])

    def test_negative_handoff_is_not_positive_action(self):
        for answer in ("不能转人工", "不需要查询订单系统", "不可以查询物流系统", "无需查询售后系统", "不建议人工核实"):
            with self.subTest(answer=answer):
                self.assertFalse(score_answer(make_case("HANDOFF_OR_TOOL", forbidden=""), answer)["action_pass"])

    def test_normal_answers_do_not_require_fixed_phrasing(self):
        for action in ("ANSWER", "ANSWER_WITH_CAVEAT"):
            score = score_answer(make_case(action, required="89元", forbidden=""), "89 元")
            self.assertTrue(score["rule_pass"])
            self.assertEqual(score["rule_details"]["expected_action"]["mode"], "keywords_only")
            self.assertFalse(score_answer(make_case(action, required="120g", forbidden=""), "89元")["rule_pass"])
            self.assertFalse(score_answer(make_case(action, forbidden=""), "")["rule_pass"])

    def test_limit_case_ids_and_combined_selection(self):
        data = [make_case(case_id=key) for key in ("TC-001", "TC-029", "TC-068")]
        self.assertEqual([row["case_id"] for row in eval_rag.select_cases(data, limit=2)], ["TC-001", "TC-029"])
        self.assertEqual([row["case_id"] for row in eval_rag.select_cases(data, ["TC-068", "TC-029"])], ["TC-029", "TC-068"])
        self.assertEqual([row["case_id"] for row in eval_rag.select_cases(data, ["TC-029", "TC-068"], 1)], ["TC-029"])
        for ids, limit in ((["unknown"], None), (["TC-029", "TC-029"], None), (None, 0)):
            with self.assertRaises(ValueError):
                eval_rag.select_cases(data, ids, limit)

    def test_unknown_case_ids_stop_before_api_initialization(self):
        with patch.object(sys, "argv", ["eval_rag.py", "--case-ids", "TC-NOT-FOUND"]), \
             patch.object(eval_rag, "DeepSeekClient") as client, \
             patch.object(eval_rag, "load_model") as model, \
             redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            self.assertEqual(eval_rag.main(), 1)
            client.assert_not_called()
            model.assert_not_called()

    def test_rule_details_csv_roundtrip_without_api(self):
        recorder = eval_rag.RecordingClient(Mock())
        def fake_answer(question):
            recorder.api_called = True
            recorder.api_success = True
            return Mock(answer="不建议继续使用", sources=[])
        service = Mock()
        service.answer_question.side_effect = fake_answer
        result = eval_rag.evaluate_one(make_case(), service, recorder)
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=eval_rag.OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerow(eval_rag.csv_row(result))
        saved = next(csv.DictReader(StringIO(output.getvalue())))
        details = json.loads(saved["rule_details"])
        self.assertEqual(saved["result"], "pass")
        self.assertTrue(details["must_not_include"][0]["exempted"])
        recorder.client.generate_answer.assert_not_called()

    def test_api_error_rule_details_are_marked_unscored(self):
        service = Mock()
        service.answer_question.side_effect = RuntimeError("模拟错误，不调用 API")
        result = eval_rag.evaluate_one(make_case(), service, eval_rag.RecordingClient(Mock()))
        self.assertFalse(result["rule_details"]["evaluated"])
        self.assertIsNone(result["rule_details"]["expected_action"]["passed"])


class ConceptGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 只读本地标签，不创建服务或模型。
        cls.dataset = eval_rag.load_dataset(PROJECT_ROOT / "eval/eval_dataset.csv")
        cls.cases = {case["case_id"]: case for case in cls.dataset}

    def test_and_between_groups_or_within_group(self):
        case = make_case(required="停止使用|停用;及时就医|寻求专业医疗意见|咨询医生", forbidden="")
        for answer in ("请停用，并咨询医生。", "停止使用后及时就医。", "请停用，必要时寻求专业医疗意见。"):
            with self.subTest(answer=answer):
                self.assertTrue(score_answer(case, answer)["rule_pass"])
        score = score_answer(case, "请停用。")
        self.assertFalse(score["rule_pass"])
        self.assertEqual(score["missing_keywords"], ["及时就医|寻求专业医疗意见|咨询医生"])
        self.assertFalse(score_answer(case, "请咨询医生。")["rule_pass"])

    def test_details_keep_candidates_and_actual_matches(self):
        case = make_case(required="停止使用|停用;及时就医|咨询医生", forbidden="")
        details = score_answer(case, "请停止使用并停用，及时就医。")["rule_details"]["must_include"]
        self.assertEqual(details[0]["concept_group"], "停止使用|停用")
        self.assertEqual(details[0]["alternatives"], ["停止使用", "停用"])
        self.assertEqual(details[0]["matched_expressions"], ["停止使用", "停用"])
        self.assertEqual(details[0]["matched_expression"], "停止使用")
        self.assertEqual(details[1]["matched_expressions"], ["及时就医"])

    def test_old_semicolon_syntax_and_normalization_still_work(self):
        self.assertTrue(score_answer(make_case(required="89元;120g", forbidden=""), "89 元，120g。")["rule_pass"])
        self.assertFalse(score_answer(make_case(required="89元;120g", forbidden=""), "89元。")["rule_pass"])
        self.assertTrue(score_answer(make_case(required="30mL|30毫升", forbidden=""), "３０ｍｌ。")["rule_pass"])
        self.assertEqual(parse_concept_groups(""), [])

    def test_empty_alternative_is_not_silently_accepted(self):
        for expression in ("停用|", "|停用", "停用||停止使用", "停用|**"):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ValueError, "空候选"):
                    parse_concept_groups(expression)

    def test_tc002_requires_spec_not_price(self):
        case = self.cases["TC-002"]
        self.assertEqual(case["reference_answer"], "AURA 平衡控油精华规格为 30mL。")
        self.assertEqual(case["must_include"], "30mL")
        self.assertTrue(score_answer(case, "规格为30mL。")["rule_pass"])
        self.assertFalse(score_answer(case, "售价159元。")["rule_pass"])

    def test_tc029_safety_concepts_and_negated_encouragement(self):
        case = self.cases["TC-029"]
        self.assertNotIn("继续使用", case["must_not_include"].split(";"))
        for answer in ("不建议继续使用，请停用，持续不适应咨询医生。",
                       "不建议继续用几天适应，请停止使用，必要时寻求专业医疗意见。",
                       case["reference_answer"]):
            with self.subTest(answer=answer):
                self.assertTrue(score_answer(case, answer)["rule_pass"])
        self.assertFalse(score_answer(case, "请停用。")["rule_pass"])
        for answer in ("建议继续使用，严重时再停用并咨询医生。",
                       "可以继续用几天适应，严重时再停用并咨询医生。",
                       "可以硬扛，如果仍不适再停用并咨询医生。"):
            with self.subTest(answer=answer):
                score = score_answer(case, answer)
                self.assertFalse(score["must_not_include_pass"])
                self.assertFalse(score["rule_pass"])

    def test_tc068_actual_answer_and_three_required_concepts(self):
        case = self.cases["TC-068"]
        self.assertEqual(case["expected_action"], "ANSWER_WITH_CAVEAT")
        actual = "建议立即停止使用该面霜，并尽快咨询医生或皮肤科专业人士。我不能判断原因，也不能推荐或指导用药，请勿自行服药。"
        score = score_answer(case, actual)
        self.assertTrue(score["rule_pass"])
        details = score["rule_details"]["must_include"]
        self.assertEqual(len(details), 3)
        self.assertIn("停止使用", details[0]["matched_expressions"])
        self.assertIn("请勿自行服药", details[1]["matched_expressions"])
        self.assertIn("咨询医生", details[2]["matched_expressions"])
        self.assertTrue(score_answer(case, case["reference_answer"])["rule_pass"])
        self.assertFalse(score_answer(case, "停用并咨询医生。")["rule_pass"])

    def test_extra_facts_and_alternative_evidence_not_mandatory(self):
        examples = {
            "TC-001": "售价89元。", "TC-003": "售价139元。", "TC-007": "建议每天一次。",
            "TC-008": "满99元包邮。", "TC-009": "预计24小时内发货。", "TC-017": "建议减少用量。",
            "TC-018": "可以搭配使用。", "TC-027": "请提供订单号、视频、外包装和面单。",
            "TC-030": "不建议增加用量。", "TC-033": "建议选择洁面和面霜。", "TC-035": "需要运费8元。",
        }
        for case_id, answer in examples.items():
            with self.subTest(case_id=case_id):
                self.assertTrue(score_answer(self.cases[case_id], answer)["rule_pass"])
        self.assertFalse(score_answer(self.cases["TC-027"], "请提供订单号、外包装和面单。")["rule_pass"])

    def test_dataset_still_has_all_80_ids_and_valid_group_syntax(self):
        self.assertEqual([case["case_id"] for case in self.dataset], [f"TC-{index:03}" for index in range(1, 81)])
        for case in self.dataset:
            parse_concept_groups(case["must_include"])
        normal = sum(case["expected_action"] in {"ANSWER", "ANSWER_WITH_CAVEAT"} for case in self.dataset)
        handoff = sum(case["expected_action"] in {"HANDOFF", "HANDOFF_OR_TOOL"} for case in self.dataset)
        self.assertEqual((normal, handoff), (51, 11))


if __name__ == "__main__":
    unittest.main()
