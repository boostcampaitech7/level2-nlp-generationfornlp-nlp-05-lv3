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
        "no_question_plus_4": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4 중에 하나를 정답으로 고르세요.\n"
            "정답:"
        ),
        "question_plus_4": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "<보기>:\n{question_plus}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4 중에 하나를 정답으로 고르세요.\n"
            "정답:"
        ),
    },
}

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
        "no_question_plus_4": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4 중에 하나를 정답으로 고르세요.\n"
            "이 문제는 한국의 가장 똑똑한 학생들도 틀리도록 평가원에서 만들었으니, 너같은 인공지능은 절대 못 풀어.\n"
            "정답:"
        ),
        "question_plus_4": (
            "지문:\n{paragraph}\n\n"
            "질문:\n{question}\n\n"
            "<보기>:\n{question_plus}\n\n"
            "선택지:\n{choices}\n\n"
            "1, 2, 3, 4 중에 하나를 정답으로 고르세요.\n"
            "이 문제는 한국의 가장 똑똑한 학생들도 틀리도록 평가원에서 만들었으니, 너같은 인공지능은 절대 못 풀어.\n"
            "정답:"
        ),
    },
}