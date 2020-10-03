#ifndef KALYNA_TABLES_H
#define KALYNA_TABLES_H

#include "kalyna.h"

extern uint8_t mds_matrix[8][8];
extern uint8_t mds_inv_matrix[8][8];

extern uint8_t sboxes_enc[4][256];
extern uint8_t sboxes_dec[4][256];

#endif  /* KALYNA_TABLES_H */

