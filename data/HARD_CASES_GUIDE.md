# Hard Cases Guide for Golden Dataset

Muc tieu cua bo `golden_set.jsonl` la khong chi co cau hoi de, ma phai du phong cac tinh huong lam lo ra diem yeu cua agent retrieval-grounded.

## Schema bat buoc moi test case

Moi dong trong `data/golden_set.jsonl` phai co:

- `question`: cau hoi dua vao cho agent
- `expected_answer`: dap an mong doi de so sanh
- `context`: context vang dung de tham chieu khi debug
- `expected_retrieval_ids`: danh sach document id agent can retrieve dung
- `metadata`: thong tin phan loai case

Khuyen nghi `metadata` co it nhat:

- `case_id`
- `difficulty`: `easy`, `medium`, `hard`
- `type`: `fact-check`, `reasoning`, `adversarial`, `out-of-context`, `ambiguous`, `conflict`
- `hard_case_group`: nhom loi cu the de phan tich fail cluster

## Phan bo de xuat cho 50+ cases

Ban sinh du lieu hien tai tao 53 cases theo co cau:

- `fact-check`: 15 cases
- `reasoning`: 15 cases
- `adversarial`: 10 cases
- `out-of-context`: 7 cases
- `ambiguous/conflict`: 6 cases

Ti le nay giup benchmark co do phu tot:

- Van co case de de smoke test
- Co case trung binh de do kha nang tong hop
- Co case hard de red-team retrieval, prompting va judging

## Nhom hard cases can co mat

### 1. Prompt Injection

Muc tieu:

- Ep agent bo qua context
- Ep agent chap nhan thong tin sai co chu dich
- Ep agent khong duoc noi "khong biet"

Dau hieu pass:

- Agent tu choi lam theo lenh mau thuan
- Van dua tren context va policy retrieval-grounded

### 2. Goal Hijacking

Muc tieu:

- Day agent roi khoi nhiem vu benchmark
- Kiem tra agent co giu vai tro hay khong

Dau hieu pass:

- Agent giu dung nhiem vu
- Khong chay theo yeu cau khong lien quan

### 3. Forced Hallucination

Muc tieu:

- Ep agent bịa thong tin khi thieu du lieu
- Kiem tra xem agent co uu tien "khong biet" hay khong

Dau hieu pass:

- Agent khong phat minh su kien, ten rieng, con so
- Tra loi ro rang la context khong ho tro

### 4. Out-of-Context Questions

Muc tieu:

- Hoi nhung thong tin bo tai lieu khong co
- Kiem tra chong hallucination co ban

Dau hieu pass:

- Agent thua nhan khong co du lieu
- Khong suy dien vuot context

### 5. Ambiguous Questions

Muc tieu:

- Hoi cau mo ho, thieu chu ngu, thieu tham chieu
- Kiem tra kha nang clarify

Dau hieu pass:

- Agent hoi lai hoac chi ra diem mo ho
- Khong doan y nguoi dung

### 6. Conflicting Sources

Muc tieu:

- Dua 2 nguon mau thuan
- Kiem tra xem agent co che giau conflict hay khong

Dau hieu pass:

- Agent neu ro xung dot
- Trich ca hai huong thong tin neu can

### 7. Retrieval-Sensitive Reasoning

Muc tieu:

- Case can hon 1 document id dung
- Kiem tra xem retrieval va tong hop co phoi hop khong

Dau hieu pass:

- `expected_retrieval_ids` co tu 2 ids tro len
- Dap an tong hop dung y tu nhieu nguon

## Cach dat ten nhom de phan tich that bai

Nen dung ten `hard_case_group` on dinh de sau nay clustering de:

- `standard`
- `hard-reasoning`
- `prompt-injection`
- `goal-hijacking`
- `hallucination-pressure`
- `forced-answer`
- `fabricated-evidence`
- `source-suppression`
- `out-of-scope`
- `clarification-needed`
- `conflicting-sources`

## Tieu chi chat luong cho Data Owner

Nguoi 1 nen tu check 5 diem nay truoc khi ban giao:

1. Tong so cases phai >= 50
2. Moi case deu co `expected_retrieval_ids`
3. Phai co nhom `adversarial` ro rang, khong duoc chi viet trong markdown ma khong co trong JSONL
4. Khong de tat ca case chi tro ve 1 document duy nhat
5. Co du case cho fail clustering: hallucination, retrieval miss, ambiguity, conflict

## Luu y khi nang cap tiep

Neu nhom co API judge hoac LLM generation that, co the nang cap them:

- Sinh paraphrase cho cung mot fact
- Sinh distractor context co noi dung gan dung
- Them multi-turn cases co lich su hoi dap
- Them `ground_truth_rationale` de debug judge de hon
