//
// Created by Olcay Taner YILDIZ on 9.02.2023.
//

#ifndef MATH_DISTRIBUTION_H
#define MATH_DISTRIBUTION_H

static double Z_MAX = 6.0;
static double Z_EPSILON = 0.000001;
static double CHI_EPSILON = 0.000001;
static double CHI_MAX = 99999.0;
static double LOG_SQRT_PI = 0.5723649429247000870717135;
static double I_SQRT_PI = 0.5641895835477562869480795;
static double BIGX = 200.0;
static double I_PI = 0.3183098861837906715377675;
static double F_EPSILON = 0.000001;
static double F_MAX = 9999.0;

double ex(double x);
double beta(double* x, int size);
double gammaLn(double x);
double zNormal(double z);
double zInverse(double p);
double chiSquare(double x, int freedom);
double chiSquareInverse(double p, int freedom);
double fDistribution(double F, int freedom1, int freedom2);
double fDistributionInverse(double p, int freedom1, int freedom2);
double tDistribution(double T, int freedom);
double tDistributionInverse(double p, int freedom);

#endif //MATH_DISTRIBUTION_H
