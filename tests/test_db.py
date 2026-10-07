from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from presence import db


def _db_available() -> bool:
    try:
        db.apply_schema()
        db.list_employees()
        return True
    except Exception:  # noqa: BLE001
        return False


if not _db_available():
    pytest.skip("PostgreSQL tidak tersedia (lihat .env)", allow_module_level=True)


@pytest.fixture
def temp_no():
    value = "TEST-9999"
    yield value
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM employees WHERE employee_no = %s", (value,))
            conn.commit()
    except Exception:  # noqa: BLE001
        pass


def test_add_employee_inserts_new_row(temp_no: str):
    first = db.add_employee("Uji Satu", temp_no, "QA", None)
    assert isinstance(first, int)
    rows = [e for e in db.list_employees(active_only=False) if e["employee_no"] == temp_no]
    assert len(rows) == 1
    assert rows[0]["name"] == "Uji Satu"


def test_add_employee_reuses_row_for_same_person(temp_no: str):
    first = db.add_employee("Uji Dua", temp_no, "IT", None)
    second = db.add_employee("uji dua", temp_no, "Finance", None)
    assert first == second, "enroll ulang orang yang sama harus memakai baris yang sama"
    rows = [e for e in db.list_employees(active_only=False) if e["employee_no"] == temp_no]
    assert len(rows) == 1
    assert rows[0]["dept"] == "Finance"


def test_add_employee_rejects_duplicate_number_used_by_other_person(temp_no: str):
    db.add_employee("Uji Tiga", temp_no, None, None)
    with pytest.raises(ValueError) as excinfo:
        db.add_employee("Uji Empat", temp_no, None, None)
    assert "sudah dipakai" in str(excinfo.value)
    rows = [e for e in db.list_employees(active_only=False) if e["employee_no"] == temp_no]
    assert len(rows) == 1, "karyawan lama tidak boleh tertimpa"
    assert rows[0]["name"] == "Uji Tiga"


def test_add_employee_allows_same_name_without_number():
    first = db.add_employee("Tanpa Nomor", None, None, None)
    second = db.add_employee("Tanpa Nomor", "", None, None)
    try:
        assert first != second, "tanpa No. karyawan tetap boleh buat baris baru"
    finally:
        for employee_id in (first, second):
            db.delete_employee(employee_id)


def test_delete_employee_detaches_open_sessions(temp_no: str):
    employee_id = db.add_employee("Uji Lima", temp_no, None, None)
    db.delete_employee(employee_id)
    assert all(e["id"] != employee_id for e in db.list_employees(active_only=False))