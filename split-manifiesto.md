# Manifiesto del split honesto — datos propios de Fase 1

Fijado en la planificación de Fase 1 (**2026-07-08**), ANTES de grabar nada
(DECISIONES.md de sal §16; plan_ml §3 "la validación honesta manda" y §14.1).
Este fichero se copia al sandbox de evaluación (regla c de su CLAUDE.md) y
**NO se modifica tras la primera anotación**.

## Unidad de asignación

La unidad es la **SESIÓN** (lugar + día + condición). Una sesión entera va a un
único bloque: **ninguna sesión aporta frames a dos bloques** — el error nº 1
que infla métricas (plan_ml §14.1) queda excluido por construcción.

## Asignación (verbatim de §16)

| Bloque | Sesiones |
|---|---|
| **train** | `parcela-D1-día`, `oficina-D1-día`, `parcela-D2-atardecer` |
| **validación** | `parcela-D3-día`, `oficina-D2-atardecer` — días DISTINTOS, mismas localizaciones |
| **test ciego** | `casa` ÍNTEGRA — localización jamás vista en train; **se evalúa UNA vez, al final** |

Reglas derivadas:

- El test ciego no se mira, no se tunea contra él, no se re-evalúa: una pasada
  al final de la fase. Si se contamina, deja de ser test.
- El set de validación es **propio, honesto e intocable** (§16): sirve para
  elegir modelo/umbral, nunca para entrenar ni pre-anotar el propio bloque.
- Si una grabación sale mal (cámara caída, condición equivocada), la sesión se
  repite con OTRO identificador de día; la asignación de bloques no se
  reorganiza.

## Nomenclatura de sesiones

```
sesion_{lugar}_{fecha}_{cam}_{condicion}
```

- `{lugar}`: `parcela` | `oficina` | `casa`
- `{fecha}`: `AAAAMMDD` (el "D1/D2/D3" de la tabla se materializa en fechas
  reales al grabar; D1 < D2 < D3 cronológicamente por lugar)
- `{cam}`: `cam1` | `cam2` (las dos Uniview; la sesión multivista usa ambas
  carpetas con el mismo prefijo de sesión)
- `{condicion}`: `dia` | `atardecer` (extensible: `noche`, `lluvia`, …)

Ejemplo: `sesion_parcela_20260815_cam1_dia`.

Los datos de sesión viven en `PROVIDENCE-DATA\propios\{sesion}\` bajo DVC; en
el repo solo este manifiesto y los datasheets (regla 6 de CLAUDE.md).
