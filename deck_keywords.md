# ECLIPSE Black-Oil Deck Parser Reference

> This document is focused on the core black-oil keywords commonly required for implementing an ECLIPSE-style deck parser.
> Each keyword includes purpose and a concrete example as it would appear in a deck.

# RUNSPEC

## SPECGRID

Purpose:
Defines logical grid dimensions.

Example:

```text
SPECGRID
 50 40 10 1 F /
```

Meaning:

- NX = 50
- NY = 40
- NZ = 10

---

## DIMENS

Purpose:
Alternative dimension declaration used in some decks.

Example:

```text
DIMENS
 50 40 10 /
```

---

## FIELD

Purpose:
Use field units.

Example:

```text
FIELD
```

---

## METRIC

Purpose:
Use metric units.

Example:

```text
METRIC
```

---

## LAB

Purpose:
Use laboratory units.

Example:

```text
LAB
```

---

## OIL

Purpose:
Enable oil phase.

```text
OIL
```

---

## WATER

Purpose:
Enable water phase.

```text
WATER
```

---

## GAS

Purpose:
Enable gas phase.

```text
GAS
```

---

## DISGAS

Purpose:
Enable solution gas.

```text
DISGAS
```

---

## VAPOIL

Purpose:
Enable vaporized oil.

```text
VAPOIL
```

---

## START

Purpose:
Simulation start date.

```text
START
 1 JAN 2020 /
```

---

## TABDIMS

Purpose:
Allocate table dimensions.

```text
TABDIMS
 20 20 20 20 1 20 20 20 /
```

---

## WELLDIMS

```text
WELLDIMS
 100 50 20 10 10 /
```

---

## EQLDIMS

```text
EQLDIMS
 10 10 10 10 10 /
```

---

# GRID

## COORD

Purpose:
Corner-point pillar coordinates.

```text
COORD
 0 0 1000 0 0 0
 100 0 1000 100 0 0
 200 0 1000 200 0 0
/
```

---

## ZCORN

Purpose:
Corner depths.

```text
ZCORN
 3000 3000 3001 3001
 3010 3010 3011 3011
/
```

---

## ACTNUM

Purpose:
Active cell mask.

```text
ACTNUM
 100*1
 20*0
 80*1
/
```

---

## MAPAXES

```text
MAPAXES
 0 0 1000 0 0 1000 /
```

---

## GRIDUNIT

```text
GRIDUNIT
 METRES /
```

---

## MAPUNIT

```text
MAPUNIT
 METRES /
```

---

## PINCH

```text
PINCH
 0.01 /
```

---

## PINCHOUT

```text
PINCHOUT
/
```

---

## DX

```text
DX
 20000*100 /
```

---

## DY

```text
DY
 20000*100 /
```

---

## DZ

```text
DZ
 20000*20 /
```

---

## TOPS

```text
TOPS
 20000*3000 /
```

---

## DXV

```text
DXV
 100 100 100 150 150 200 /
```

---

## DYV

```text
DYV
 100 100 150 150 200 /
```

---

## DZV

```text
DZV
 10 15 20 20 25 /
```

---

## PORO

```text
PORO
 500*0.18
 500*0.22
/
```

---

## NTG

```text
NTG
 1000*0.85 /
```

---

## PERMX

```text
PERMX
 1000*500 /
```

---

## PERMY

```text
PERMY
 1000*300 /
```

---

## PERMZ

```text
PERMZ
 1000*50 /
```

---

## MULTX

```text
MULTX
 1000*0.5 /
```

---

## MULTX-

```text
MULTX-
 1000*0.25 /
```

---

## MULTY

```text
MULTY
 1000*0.75 /
```

---

## MULTY-

```text
MULTY-
 1000*0.75 /
```

---

## MULTZ

```text
MULTZ
 1000*1.2 /
```

---

## MULTZ-

```text
MULTZ-
 1000*1.2 /
```

---

## FAULTS

```text
FAULTS
 'F1' 10 10 1 40 1 10 X /
 'F2' 25 25 1 40 1 10 Y /
/
```

---

## MULTFLT

```text
MULTFLT
 'F1' 0.01 /
 'F2' 0.10 /
/
```

---

## NNC

```text
NNC
 10 10 3 15 10 3 25.0 /
 10 10 4 15 10 4 25.0 /
/
```

---

# EDIT

## BOX

```text
BOX
 1 20 1 20 1 5 /
```

---

## ENDBOX

```text
ENDBOX
```

---

## EQUALS

```text
EQUALS
 PORO 0.25 /
/
```

Variation:

```text
EQUALS
 PORO 0.25 1 20 1 20 1 5 /
/
```

---

## ADD

```text
ADD
 PORO 0.02 /
/
```

---

## COPY

```text
COPY
 PERMX PERMY /
/
```

---

## MULTIPLY

```text
MULTIPLY
 PERMX 0.1 /
/
```

---

# PROPS

## DENSITY

```text
DENSITY
 53.0 64.0 0.06 /
```

---

## PVTW

```text
PVTW
 4000 1.02 3.0E-6 0.3 0.0 /
/
```

---

## PVDO

```text
PVDO
 1000 1.20 2.0
 2000 1.18 1.8
 3000 1.16 1.6
/
```

---

## PVCO

```text
PVCO
 3000 1.15 1.0E-5 1.8 0.0 /
/
```

---

## PVTO

```text
PVTO
 200
 1000 1.35 0.8
 3000 1.20 1.5
/
```

---

## PVDG

```text
PVDG
 500 0.005 0.02
 1000 0.004 0.02
 2000 0.003 0.03
/
```

---

## PVTG

```text
PVTG
 500
 0.00 0.005 0.02
 0.05 0.006 0.03
/
```

---

## ROCK

```text
ROCK
 3000 5.0E-6 /
/
```

---

## ROCKTAB

```text
ROCKTAB
 1000 1.00
 3000 0.99
 5000 0.98
/
```

---

## SWOF

```text
SWOF
 0.20 0.00 1.00 20
 0.30 0.05 0.80 15
 0.50 0.30 0.50 8
 1.00 1.00 0.00 0
/
```

---

## SGOF

```text
SGOF
 0.00 0.00 1.00 0
 0.10 0.05 0.80 5
 0.30 0.30 0.40 15
 1.00 1.00 0.00 25
/
```

---

## SWFN

```text
SWFN
 0.20 0.00 20
 0.50 0.30 8
 1.00 1.00 0
/
```

---

## SGFN

```text
SGFN
 0.00 0.00 0
 0.30 0.30 15
 1.00 1.00 25
/
```

---

## SOF2

```text
SOF2
 0.20 1.00
 0.50 0.50
 1.00 0.00
/
```

---

## SOF3

```text
SOF3
 0.20 1.00
 0.50 0.50
 1.00 0.00
/
```

---

# REGIONS

## SATNUM

```text
SATNUM
 1000*1 /
```

---

## PVTNUM

```text
PVTNUM
 500*1 500*2 /
```

---

## EQLNUM

```text
EQLNUM
 1000*1 /
```

---

## FIPNUM

```text
FIPNUM
 1000*1 /
```

---

## ROCKNUM

```text
ROCKNUM
 1000*1 /
```

---

## IMBNUM

```text
IMBNUM
 1000*1 /
```

---

# SOLUTION

## PRESSURE

```text
PRESSURE
 1000*3500 /
```

---

## SWAT

```text
SWAT
 1000*0.25 /
```

---

## SGAS

```text
SGAS
 1000*0.05 /
```

---

## SOIL

```text
SOIL
 1000*0.70 /
```

---

## RS

```text
RS
 1000*500 /
```

---

## RV

```text
RV
 1000*0.01 /
```

---

## EQUIL

```text
EQUIL
 7000 3500
 7200 0
 6800 0
 0 0
/
```

---

## RESTART

```text
RESTART
 CASE1 100 /
```

---

# SUMMARY

## Field Vectors

```text
FOPR
FWPR
FGPR
FOPT
FWPT
FGPT
```

## Well Vectors

```text
WOPR
WWPR
WGPR
WBHP
WTHP
```

## Region Vectors

```text
ROIP
RGIP
RWIP
```

## Reporting

```text
RPTRST
 BASIC=2 /

RPTSCHED
 BASIC=2 /
```

---

# SCHEDULE

## DATES

```text
DATES
 1 JAN 2020 /
 1 FEB 2020 /
/
```

---

## TSTEP

```text
TSTEP
 30 30 30 90 /
```

---

## WELSPECS

```text
WELSPECS
 'PROD1' 'FIELD' 10 10 OIL /
 'INJ1'  'FIELD' 30 30 WATER /
/
```

---

## COMPDAT

```text
COMPDAT
 'PROD1' 10 10 1 5 OPEN 1* 1*
          0.311 1000 0 1* Z /
/
```

---

## WCONPROD

```text
WCONPROD
 'PROD1' OPEN ORAT 5000 4* 2500 /
/
```

---

## WCONINJE

```text
WCONINJE
 'INJ1' WATER OPEN RATE 3000 5000 /
/
```

---

## WELOPEN

```text
WELOPEN
 'PROD1' SHUT /
/
```

---

## WELTARG

```text
WELTARG
 'PROD1' ORAT 7000 /
/
```

---

## WPIMULT

```text
WPIMULT
 'PROD1' 0.5 /
/
```

---

## GRUPTREE

```text
GRUPTREE
 'PROD' 'FIELD' /
 'INJ'  'FIELD' /
/
```

---

## GCONPROD

```text
GCONPROD
 'PROD' ORAT 20000 /
/
```

---

## GCONINJE

```text
GCONINJE
 'INJ' WATER RATE 15000 /
/
```

---

## WECON

```text
WECON
 'PROD1' 10 1000 1* 1* SHUT /
/
```

---

## WTEST

```text
WTEST
 30 /
```

---

# END

```text
END
```
