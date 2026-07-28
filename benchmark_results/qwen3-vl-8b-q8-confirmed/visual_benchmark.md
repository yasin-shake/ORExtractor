# Visual-model benchmark

- Provider: `ollama`
- Model: `qwen3-vl:8b-instruct-q8_0`

| Metric | Result |
|---|---:|
| Total cases | 28 |
| Gold cases | 8 |
| Successful/schema-valid | 28 |
| Gold pass rate | 100.00% |
| Schema-valid rate | 100.00% |
| Classification accuracy | 100.00% |
| Numeric recall | 100.00% |
| Mean model latency | 5631.7 ms |

## Cases

| Case | Source | Task | Schema | Gold pass | Latency (ms) | Error |
|---|---|---|---:|---:|---:|---|
| resource-table | synthetic | table | True | True | 2739.1 |  |
| economics-table | synthetic | table | True | True | 2264.5 |  |
| production-bar-chart | synthetic | figure | True | True | 5949.7 |  |
| recovery-line-chart | synthetic | figure | True | True | 5499.7 |  |
| grade-scatter-chart | synthetic | figure | True | True | 5944.6 |  |
| process-flow-diagram | synthetic | figure | True | True | 5728.4 |  |
| geological-cross-section | synthetic | figure | True | True | 2956.6 |  |
| mine-plan | synthetic | figure | True | True | 2943.1 |  |
| Sedar_2024 NI 43 101 Technical Report/February 2024/18_Rokmaster Resources Corp.  Rokmaster Resources Corp. (000031923).pdf:p133:ec62767533ed180f9a46e43d | retained-artifact | table | True | n/a | 5307.0 |  |
| EDV-SMGO-TR01-NRP-002_0_Sabodala_MC_REP_1_SEDAR.pdf:p588:90604457d7ca3387af1c569c | retained-artifact | table | True | n/a | 6975.4 |  |
| Sedar_2024 NI 43 101 Technical Report/March 2024/1_Temas Resources Corp.  Temas Resources Corp. (000049522).pdf:p133:42ea141f535851bc36363789 | retained-artifact | table | True | n/a | 1359.8 |  |
| Sedar_2024 NI 43 101 Technical Report/June_2024/2_Fury Gold Mines Limited  Fury Gold Mines Limited (000027371) 27 Jun 2024 2106 EDT.pdf:p18:1fb14d094338a242420f8cf9 | retained-artifact | table | True | n/a | 6012.9 |  |
| KatangaMining_Technical_report_11072019.pdf:p170:a327d7e6878808fac7e733f8 | retained-artifact | table | True | n/a | 13219.0 |  |
| Sedar_2024 NI 43 101 Technical Report/February 2024/15_Montage Gold Corp.  Montage Gold Corp. (000048267).pdf:p215:bb59ccbaeeaec5f5d2015212 | retained-artifact | table | True | n/a | 3485.1 |  |
| Sedar_2024 NI 43 101 Technical Report/February 2024/11_Andean Precious Metals Corp.  Andean Precious Metals Corp. (000047056).pdf:p175:d42b536f750391b283be5fa4 | retained-artifact | table | True | n/a | 5816.1 |  |
| Sedar_2024 NI 43 101 Technical Report/November 2024/6_Prime Mining Corp.  Prime Mining Corp. (000005846).pdf:p254:230b8015806f65b924612810 | retained-artifact | table | True | n/a | 4448.1 |  |
| Technical Assessment Report.pdf:p92:f20f9baf90670eb211515233 | retained-artifact | table | True | n/a | 2028.5 |  |
| Sedar_2024 NI 43 101 Technical Report/August 2024/22_Westgold Resources Limited  Westgold Resources Limited (000047807) 01 Aug 2024 1213 EDT.pdf:p283:8396f30c2d39c91ede9b52b8 | retained-artifact | table | True | n/a | 2788.0 |  |
| Sedar_2024 NI 43 101 Technical Report/January 2024/1_Andean Precious Metals Corp.pdf:p277:2e156bbdff476be6ec4772ae | retained-artifact | table | True | n/a | 4295.2 |  |
| Sedar_2024 NI 43 101 Technical Report/March 2024/17_Trigon Metals Inc.  Trigon Metals Inc. (000010678).pdf:p223:2475537b34c781c7d5c6bfc1 | retained-artifact | table | True | n/a | 10187.9 |  |
| R_05-2898-01_SRK_ABR_20260123_RevD.pdf:p168:d4e7e71a0acbeed2a469c9cf | retained-artifact | table | True | n/a | 5565.8 |  |
| ncu_ni_43-101_report_16apr2019.pdf:p180:1c07cde3d9c278d6df7da9c5 | retained-artifact | table | True | n/a | 3684.2 |  |
| G10129-RPX_PEA_Rep_Rev_D-Final.pdf:p196:78e8f5f015b0d5506dbfe93a | retained-artifact | table | True | n/a | 17029.8 |  |
| Form_45-106F1_Report_of_Exempt_Distribution_for_UBS_(Canada)_Global_Master_Fund___UBS_(Canada)_Global_Master_Fund_(07_Jun_2026)_-_EN.pdf:p4:bd1731b15a15f903fd16c12c | retained-artifact | table | True | n/a | 3223.2 |  |
| technical_report_-_Moss_Mine_MRE.pdf:p156:4cfb092249e45d5ee2db11f7 | retained-artifact | table | True | n/a | 12263.8 |  |
| 7._Technical_Report.pdf:p98:1d706e3cdd5826ac1aa07dec | retained-artifact | table | True | n/a | 2868.3 |  |
| Aura_Matupa_SK1300_GE21_final_.pdf:p225:e6d65be35b676cab2a1908d8 | retained-artifact | table | True | n/a | 6890.5 |  |
| Sedar_2024 NI 43 101 Technical Report/October 2024/35_WINSOME RESOURCES LTD.  WINSOME RESOURCES LTD. (000056341).pdf:p172:1344135bc7ad211684e28be9 | retained-artifact | table | True | n/a | 6212.5 |  |
