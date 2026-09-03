-- Additive migration — run this once against your Supabase project.
-- Adds tables for Quiz Studio (POST /materials/{id}/quiz,
-- /quiz/{quiz_id}/attempt, GET /materials/{id}/quiz-attempts), which the
-- frontend already calls but had no backend support. Nothing here alters
-- or drops any existing table.

create table if not exists material_quizzes (
    id uuid primary key default gen_random_uuid(),
    material_id uuid not null references materials(id) on delete cascade,
    student_id uuid not null references auth.users(id) on delete cascade,
    num_questions integer not null,
    difficulty text not null,
    marking text not null,
    -- Full question objects including correct_index/explanation — never
    -- returned as-is to the browser (see app/materials.py's public_questions
    -- stripping in generate_material_quiz). Only submit_quiz_attempt reads
    -- correct_index/explanation back out, server-side, to grade.
    questions jsonb not null,
    created_at timestamptz not null default now()
);
alter table material_quizzes enable row level security;
create index if not exists material_quizzes_material_id_idx
    on material_quizzes(material_id, student_id);


create table if not exists material_quiz_attempts (
    id uuid primary key default gen_random_uuid(),
    quiz_id uuid not null references material_quizzes(id) on delete cascade,
    material_id uuid not null references materials(id) on delete cascade,
    student_id uuid not null references auth.users(id) on delete cascade,
    answers jsonb not null default '{}'::jsonb,
    score numeric not null,
    max_score numeric not null,
    correct_count integer not null,
    wrong_count integer not null,
    unanswered_count integer not null,
    time_taken_seconds integer,
    created_at timestamptz not null default now()
);
alter table material_quiz_attempts enable row level security;
create index if not exists material_quiz_attempts_material_id_idx
    on material_quiz_attempts(material_id, student_id, created_at);

-- No RLS policies added — same pattern as the rest of this schema: the
-- backend talks to Supabase with the service-role key (bypasses RLS), so
-- leaving these tables policy-less just means no anon/direct client can
-- read or write them, which is the safe default here.
