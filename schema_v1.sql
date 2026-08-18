-- Run this in Supabase SQL Editor (Project -> SQL Editor -> New query)

create extension if not exists "pgcrypto";

-- One row per teacher persona
create table if not exists personas (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    source_youtube_url text,
    status text not null default 'processing'
        check (status in ('processing', 'ready', 'failed')),
    error_message text,

    teaching_style text,
    topics_covered text,
    problem_solving_approach text,
    solved_questions text,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- One row per student<->persona chat thread
create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    persona_id uuid not null references personas(id) on delete cascade,
    student_id uuid not null references auth.users(id) on delete cascade,
    created_at timestamptz not null default now()
);

-- One row per message (student question or tutor answer)
create table if not exists messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references conversations(id) on delete cascade,
    role text not null check (role in ('student', 'tutor')),
    content text not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_conversations_persona on conversations(persona_id);
create index if not exists idx_conversations_student on conversations(student_id);
create index if not exists idx_messages_conversation on messages(conversation_id);

-- Row Level Security: students can only see their own conversations/messages.
-- The backend uses the SERVICE ROLE key (bypasses RLS) for all writes, so
-- these policies mainly matter if your frontend ever queries Supabase
-- directly with the student's own anon/JWT session.
alter table conversations enable row level security;
alter table messages enable row level security;

create policy "Students see their own conversations"
    on conversations for select
    using (auth.uid() = student_id);

create policy "Students see messages in their own conversations"
    on messages for select
    using (
        conversation_id in (
            select id from conversations where student_id = auth.uid()
        )
    );

-- personas table is public read (students need to browse/select a tutor)
alter table personas enable row level security;
create policy "Anyone can view ready personas"
    on personas for select
    using (status = 'ready');
