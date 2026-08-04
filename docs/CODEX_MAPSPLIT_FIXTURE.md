수정:

- [scripts/uv_mapsplit_offline.py:510](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_mapsplit_offline.py:510): 서로 다른 유한 양수 `eps_l` 6개와 `eps_phys=1` 추가.
- [scripts/uv_mapsplit_offline.py:519](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_mapsplit_offline.py:519): `eps_phys=0` fixture 추가. energy/noeps의 바이트 동일성을 검사하고, `eps_phys=1`에서는 상이함을 검사.

정상 자기시험(exit 0):

```text
"eps_phys_0_energy_bitwise_identical_to_noeps": true
"eps_phys_1_energy_differs_from_noeps": true
"status": "PASS"
```

`eta = w * S_l_used` 결함 주입 자기시험(exit 2):

```text
UNRESOLVED-FAIL-CLOSED: eps_phys=1 fixture energy did not differ from noeps
```

주입 결함은 제거했고 production 식은 원래대로 복구했습니다. 남은 미해결 사항은 없습니다.