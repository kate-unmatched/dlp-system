from sqlmodel import SQLModel, Field

class UserRole(SQLModel, table=True):
    """
    Промежуточная таблица связи “многие‑ко‑многим” между User и Role.
    Composite PK: (user_id, role_id)
    """
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    role_id: int = Field(foreign_key="role.id", primary_key=True)
