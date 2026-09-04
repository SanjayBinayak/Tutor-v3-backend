-- Additive migration — run this once against your Supabase project, after
-- schema_v1 through schema_v6. Nothing here alters or drops any existing
-- table, it only adds columns/tables.
--
-- Adds:
--   1. conversations.title / conversations.updated_at — lets
--      GET /conversations return a ChatGPT-style history list (title +
--      last-active time) without a separate table.
--   2. persona_quizzes + persona_quiz_attempts — Quiz section for a
--      persona (tutor), mirroring the old material_quizzes /
--      material_quiz_attempts pattern (schema_v4_quiz.sql) but built from
--      a persona's teaching material instead of an uploaded document.

alter table conversations
    add column if not exists title text,
    add column if not exists updated_at timestamptz not null default now();

create index if not exists idx_conversations_student_updated
    on conversations(student_id, updated_at desc);


create table if not exists persona_quizzes (
    id uuid primary key default gen_random_uuid(),
    persona_id uuid not null references personas(id) on delete cascade,
    student_id uuid not null references auth.users(id) on delete cascade,
    num_questions integer not null,
    difficulty text not null,
    marking text not null,
    -- Full question objects including correct_index/explanation — never
    -- returned as-is to the browser. Only the attempt endpoint reads
    -- correct_index/explanation back out, server-side, to grade.
    questions jsonb not null,
    created_at timestamptz not null default now()
);
alter table persona_quizzes enable row level security;
create index if not exists persona_quizzes_persona_id_idx
    on persona_quizzes(persona_id, student_id);


create table if not exists persona_quiz_attempts (
    id uuid primary key default gen_random_uuid(),
    quiz_id uuid not null references persona_quizzes(id) on delete cascade,
    persona_id uuid not null references personas(id) on delete cascade,
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
alter table persona_quiz_attempts enable row level security;
create index if not exists persona_quiz_attempts_persona_id_idx
    on persona_quiz_attempts(persona_id, student_id, created_at);

-- No RLS policies added — same pattern as the rest of this schema: the
-- backend talks to Supabase with the service-role key (bypasses RLS), so
-- leaving these tables policy-less just means no anon/direct client can
-- read or write them, which is the safe default here.
