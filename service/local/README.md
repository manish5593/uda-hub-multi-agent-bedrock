# Your own demo customers

To run the demo as yourself, describe yourself in `customers.json` in this folder. Git ignores that file,
so your name and email never reach the repository. The seed job (`python -m service.scripts.seed`, run by
`docker compose up`) adds every customer listed there that isn't in the databases yet. That happens after
the committed demo data is loaded, so you survive `seed --reset`.

```json
[
  {
    "user_id": "1a2b3c",
    "full_name": "Your Name",
    "email": "you@example.com",
    "subscription": {"subscription_id": "4d5e6f", "status": "active", "tier": "premium", "monthly_quota": 4},
    "reservations": [{"reservation_id": "7a8b9c", "experience_id": "8aaed6"}],
    "password_hash": "$2b$12$..."
  }
]
```

- **Ids** are 6 hex characters with at least one digit and one letter, like the demo data's, and must not
  clash with it. `experience_id` must be one of the demo experiences (`8aaed6` is Modern Art at MASP).
- **`password_hash`** is optional. Without it, sign up as usual (`/v1/auth/register/start`); the login is
  kept in the auth database, which `seed --reset` doesn't touch. With it, the login is recreated whenever the
  auth database is new, for example after `docker compose down -v`. Copy it from the auth database after
  signing up once; it is a bcrypt hash, never the password:

  ```bash
  docker compose exec db psql -U postgres -d auth -tAc "select password_hash from credentials where email = 'you@example.com'"
  ```

The file is mounted into the seed job only, read-only, and kept out of the image (`.dockerignore`).
