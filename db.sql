-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table (extends Supabase auth.users)
create table public.profiles (
  id uuid references auth.users on delete cascade not null primary key,
  updated_at timestamp with time zone,
  username text unique,
  full_name text,
  avatar_url text,
  
  constraint username_length check (char_length(username) >= 3)
);

-- Create sources table
create table public.sources (
  id uuid default uuid_generate_v4() primary key,
  user_id uuid references auth.users not null,
  name text not null,
  type text not null check (type in ('pdf', 'video', 'image', 'text', 'url')),
  file_path text,
  size bigint,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
  metadata jsonb default '{}'::jsonb,
  
  constraint valid_file_size check (size > 0)
);

-- Create RLS policies
alter table public.profiles enable row level security;
alter table public.sources enable row level security;

-- Profiles policies
create policy "Public profiles are viewable by everyone."
  on profiles for select
  using ( true );

create policy "Users can insert their own profile."
  on profiles for insert
  with check ( auth.uid() = id );

create policy "Users can update own profile."
  on profiles for update
  using ( auth.uid() = id );

-- Sources policies
create policy "Sources are viewable by owner."
  on sources for select
  using ( auth.uid() = user_id );

create policy "Sources can be inserted by owner."
  on sources for insert
  with check ( auth.uid() = user_id );

create policy "Sources can be updated by owner."
  on sources for update
  using ( auth.uid() = user_id );

create policy "Sources can be deleted by owner."
  on sources for delete
  using ( auth.uid() = user_id );

-- Functions
create function public.handle_new_user() 
returns trigger as $$
begin
  insert into public.profiles (id, full_name, avatar_url)
  values (new.id, new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'avatar_url');
  return new;
end;
$$ language plpgsql security definer;

-- Trigger for new user creation
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();