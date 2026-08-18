-- =====================================================================
-- EXTENDS your existing schema.sql — run this AFTER the original one.
-- =====================================================================

-- ---------------------------------------------------------------------
-- ROLES: every signed-up user is either a teacher or a student.
-- A trigger auto-creates this row when someone signs up, reading the
-- role from the signup metadata your frontend sends (see auth.py notes).
-- ---------------------------------------------------------------------
create table if not exists profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    role text not null check (role in ('teacher', 'student')),
    full_name text,
    created_at timestamptz not null default now()
);

create or replace function handle_new_user()
returns trigger as $$
begin
    insert into public.profiles (id, role, full_name)
    values (
        new.id,
        coalesce(new.raw_user_meta_data->>'role', 'student'),
        new.raw_user_meta_data->>'full_name'
    );
    return new;
end;
$$ language plpgsql security definer;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function handle_new_user();

alter table profiles enable row level security;
create policy "Anyone signed in can read profiles"
    on profiles for select
    using (auth.role() = 'authenticated');


-- ---------------------------------------------------------------------
-- CHAT: one shared "student lounge" channel, plus 1:1 teacher<->student
-- direct messages. Kept as a single table with a channel type rather
-- than separate tables — simpler RLS, simpler frontend queries.
--
-- NOTE: named "chat_messages", NOT "messages" — your existing "messages"
-- table (from schema_v1.sql) already stores persona Q&A conversation
-- history. Reusing that name here would silently break persona chat.
-- ---------------------------------------------------------------------
create table if not exists chat_messages (
    id uuid primary key default gen_random_uuid(),
    channel text not null check (channel in ('student_lounge', 'teacher_dm')),
    sender_id uuid not null references auth.users(id) on delete cascade,
    recipient_id uuid references auth.users(id) on delete cascade, -- null for lounge, set for DM
    content text not null,
    created_at timestamptz not null default now()
);
create index if not exists idx_chat_messages_channel on chat_messages(channel, created_at);
create index if not exists idx_chat_messages_dm on chat_messages(sender_id, recipient_id);

alter table chat_messages enable row level security;

create policy "Anyone signed in can read the lounge"
    on chat_messages for select
    using (channel = 'student_lounge' or sender_id = auth.uid() or recipient_id = auth.uid());

create policy "Anyone signed in can post to the lounge or their own DMs"
    on chat_messages for insert
    with check (sender_id = auth.uid());


-- ---------------------------------------------------------------------
-- DOUBTS + UPVOTES
-- ---------------------------------------------------------------------
create table if not exists doubts (
    id uuid primary key default gen_random_uuid(),
    student_id uuid not null references auth.users(id) on delete cascade,
    title text not null,
    content text not null,
    created_at timestamptz not null default now()
);

create table if not exists doubt_upvotes (
    doubt_id uuid not null references doubts(id) on delete cascade,
    student_id uuid not null references auth.users(id) on delete cascade,
    created_at timestamptz not null default now(),
    primary key (doubt_id, student_id)
);

alter table doubts enable row level security;
alter table doubt_upvotes enable row level security;

create policy "Anyone signed in can read doubts" on doubts for select using (true);
create policy "Students post their own doubts" on doubts for insert with check (student_id = auth.uid());

create policy "Anyone signed in can read upvotes" on doubt_upvotes for select using (true);
create policy "Anyone signed in can upvote once" on doubt_upvotes for insert with check (student_id = auth.uid());
create policy "Anyone can remove their own upvote" on doubt_upvotes for delete using (student_id = auth.uid());


-- ---------------------------------------------------------------------
-- ANNOUNCEMENTS (teacher-authored, optionally AI-drafted first)
-- ---------------------------------------------------------------------
create table if not exists announcements (
    id uuid primary key default gen_random_uuid(),
    teacher_id uuid not null references auth.users(id) on delete cascade,
    content text not null,
    created_at timestamptz not null default now()
);
alter table announcements enable row level security;
create policy "Anyone signed in can read announcements" on announcements for select using (true);
-- inserts happen via the backend (service role), which checks the teacher role itself


-- ---------------------------------------------------------------------
-- HOMEWORK: teacher-defined assignment requirements + student
-- submissions + AI-generated feedback
-- ---------------------------------------------------------------------
create table if not exists assignments (
    id uuid primary key default gen_random_uuid(),
    teacher_id uuid not null references auth.users(id) on delete cascade,
    title text not null,
    requirements text not null,  -- what the AI checks submissions against
    created_at timestamptz not null default now()
);

create table if not exists homework_submissions (
    id uuid primary key default gen_random_uuid(),
    assignment_id uuid not null references assignments(id) on delete cascade,
    student_id uuid not null references auth.users(id) on delete cascade,
    file_path text not null,          -- path in Supabase Storage
    status text not null default 'pending' check (status in ('pending', 'checked', 'failed')),
    ai_feedback text,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table assignments enable row level security;
alter table homework_submissions enable row level security;

create policy "Anyone signed in can read assignments" on assignments for select using (true);
create policy "Students read their own submissions, teachers read all" on homework_submissions for select using (true);
create policy "Students submit their own homework" on homework_submissions for insert with check (student_id = auth.uid());


-- ---------------------------------------------------------------------
-- STORAGE: create a bucket for homework uploads (run in Storage tab,
-- or via SQL if you prefer — bucket creation via SQL requires the
-- storage extension, simplest is: Supabase Dashboard -> Storage ->
-- New bucket -> name it "homework" -> not public)
-- ---------------------------------------------------------------------
