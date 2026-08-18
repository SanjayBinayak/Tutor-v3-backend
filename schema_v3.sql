
create table if not exists persona_requests (
    id uuid primary key default gen_random_uuid(),
    requested_by_name text not null,
    requested_by_email text not null,
    teacher_name text not null,
    youtube_url text,
    notes text,
    created_at timestamptz not null default now()
);

alter table persona_requests enable row level security;



create table if not exists assignments (
    id uuid primary key default gen_random_uuid(),
    teacher_id uuid not null references auth.users(id) on delete cascade,
    title text not null,
    requirements text not null,
    created_at timestamptz not null default now()
);
alter table assignments enable row level security;

create table if not exists homework_submissions (
    id uuid primary key default gen_random_uuid(),
    assignment_id uuid not null references assignments(id) on delete cascade,
    file_path text not null,
    status text not null default 'pending' check (status in ('pending', 'checked', 'failed')),
    ai_feedback text,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
alter table homework_submissions enable row level security;


alter table homework_submissions drop column if exists student_id;
alter table homework_submissions add column if not exists student_name text not null default '';
alter table homework_submissions add column if not exists student_email text not null default '';


drop policy if exists "Students submit their own homework" on homework_submissions;
drop policy if exists "Students read their own submissions, teachers read all" on homework_submissions;
drop policy if exists "Anyone signed in can read assignments" on assignments;
