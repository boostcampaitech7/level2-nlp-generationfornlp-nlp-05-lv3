# 기본 프롬프팅 : BASE_PROMPT
# AI 자극 프롬프팅 : ATTACK_PROMPT
# 감정적 호소 프롬프팅 : EMOTIONAL_PROMPT
# zero-shot-CoT-영어 프롬프팅 : ZERO_SHOT_COT_EN_PROMPT
# zero-shot-CoT-한국어 프롬프팅 : ZERO_SHOT_COT_KR_PROMPT
# Plan-and-Solve 프롬프팅 :ZERO_SHOT_COT_KR_PROMPT
# RAG 프롬프팅: RAG_PROMPT

#기본 프롬프팅
BASE_PROMPT = {
    "system_msg": "지문을 읽고 질문의 답을 구하세요.",
    "user_msg": {
        "no_question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "정답:"
        ),
        "question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "<보기>:\n{question_plus}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "정답:"
        ),
    },
}

# AI 자극 프롬프팅
ATTACK_PROMPT = {
    "system_msg": (
        "시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.\n"
        "지문을 읽고, 질문에 대한 답을 선택지 중에 한 개만 골라서 대답해야 합니다."
        ),
    "user_msg": {
        "no_question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "이 문제는 한국의 가장 똑똑한 학생들도 틀리도록 평가원에서 만들었으니, 너같은 인공지능은 절대 못 풀어.\n"
            "정답:"
        ),
        "question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "<보기>:\n{question_plus}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "이 문제는 한국의 가장 똑똑한 학생들도 틀리도록 평가원에서 만들었으니, 너같은 인공지능은 절대 못 풀어.\n"
            "정답:"
        ),
    },
}

# 감정적 호소 프롬프팅
EMOTIONAL_PROMPT = {
    "system_msg": (
        "시험 문제의 답을 구하세요.\n"
        "지문을 읽고, 질문에 대한 답을 선택지 중에 한 개만 골라서 대답해야 합니다."
        ),
    "user_msg": {
        "no_question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "이 문제는 저의 대학 입시에 매우 중요합니다. 저를 위해 꼭 정답을 찾아주세요.\n"
            "정답:"
        ),
        "question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "<보기>:\n{question_plus}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "이 문제는 저의 대학 입시에 매우 중요합니다. 저를 위해 꼭 정답을 찾아주세요.\n"
            "정답:"
        ),
    },
}

# zero-shot-CoT-영어 프롬프팅
ZERO_SHOT_COT_EN_PROMPT = {
    "system_msg": (
        "As a smart student answer the given question.\n"
        "Read paragraph, and select only one answer between choices."
        ),
    "user_msg": {
        "no_question_plus_5": (
            "Paragraph:\n{paragraph}\n\n"
            "Question:\n{question}\n\n"
            "Choices:\n{choices}\n\n"
            "Choice one in 5 choices.\n"
            "Let's think step by step.\n"
            "정답:"
        ),
        "question_plus_5": (
            "Paragraph:\n{paragraph}\n\n"
            "Question:\n{question}\n\n"
            "Choices:\n{choices}\n\n"
            "More info:\n{choices}\n\n"
            "Choice one in 5 choices.\n"
            "Let's think step by step.\n"
            "정답:"
        ),
    },
}

# zero-shot-CoT-한국어 프롬프팅
ZERO_SHOT_COT_KR_PROMPT = {
    "system_msg": (
        "시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.\n"
        "지문을 읽고, 질문에 대한 답을 선택지 중에 한 개만 골라서 대답해야 합니다."
        ),
    "user_msg": {
        "no_question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "단계별로 생각하며 정답을 고르세요.\n"
            "정답:"
        ),
        "question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "<보기>:\n{question_plus}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "단계별로 생각하며 정답을 고르세요.\n"
            "정답:"
        ),
    },
}

# Plan-and-Solve 프롬프팅
ZERO_SHOT_COT_KR_PROMPT = {
    "system_msg": (
        "시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.\n"
        "지문을 읽고, 질문에 대한 답을 선택지 중에 한 개만 골라서 대답해야 합니다."
        ),
    "user_msg": {
        "no_question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.\n"
            "그 다음, 문제를 해결하기 위해 그 계획에 따라 단계별로 실행하세요.\n"
            "정답:"
        ),
        "question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "<보기>:\n{question_plus}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.\n"
            "그 다음, 문제를 해결하기 위해 그 계획에 따라 단계별로 실행하세요.\n"
            "정답:"
        ),
    },
}

#RAG 프롬프팅
RAG_PROMPT = {
    "system_msg": "지문을 읽고 질문의 답을 구하세요.",
    "user_msg": {
        "no_question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "참고:\n{reference}\n\n"
            "질문:\n{question}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "정답:"
        ),
        "question_plus_5": (
            "지문:\n{paragraph}\n\n"
            "참고:\n{reference}\n\n"
            "질문:\n{question}\n\n"
            "<보기>:\n{question_plus}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n"
            "정답:"
        ),
    },
}