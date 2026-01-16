export interface OnboardingPayload {
  age: number;
  height: number;
  weight: number;
  cycleRegularity: "Regular" | "Irregular";
  cycleLength: number;
  missedPeriods: number;
  heavyPeriods: boolean;
  acneSeverity: "None" | "Mild" | "Severe";
  excessHair: boolean;
  hairLoss: boolean;
  suddenWeightGain: boolean;
  darkPatches: boolean;
  familyHistory: boolean;
  thyroidOrDiabetes: boolean;
  activityLevel: "Low" | "Moderate" | "High";
  sleepHours: number;
  stressLevel: 1 | 2 | 3 | 4 | 5;
}

export const calculateBmi = (heightCm: number, weightKg: number) => {
  const heightM = heightCm / 100;
  const bmi = weightKg / (heightM * heightM);
  return Math.round(bmi * 100) / 100;
};

const isNumber = (value: unknown) =>
  typeof value === "number" && !Number.isNaN(value);
const isBoolean = (value: unknown) => typeof value === "boolean";

export const validateOnboardingPayload = (payload: any) => {
  const errors: string[] = [];

  const requiredFields: Array<keyof OnboardingPayload> = [
    "age",
    "height",
    "weight",
    "cycleRegularity",
    "cycleLength",
    "missedPeriods",
    "heavyPeriods",
    "acneSeverity",
    "excessHair",
    "hairLoss",
    "suddenWeightGain",
    "darkPatches",
    "familyHistory",
    "thyroidOrDiabetes",
    "activityLevel",
    "sleepHours",
    "stressLevel",
  ];

  requiredFields.forEach((field) => {
    if (payload?.[field] === undefined || payload?.[field] === null) {
      errors.push(`${field} is required`);
    }
  });

  if (payload?.age !== undefined && !isNumber(payload.age)) {
    errors.push("age must be a number");
  } else if (
    isNumber(payload?.age) &&
    (payload.age < 10 || payload.age > 100)
  ) {
    errors.push("age must be between 10 and 100");
  }

  if (payload?.height !== undefined && !isNumber(payload.height)) {
    errors.push("height must be a number");
  } else if (
    isNumber(payload?.height) &&
    (payload.height < 50 || payload.height > 250)
  ) {
    errors.push("height must be between 50 and 250 cm");
  }

  if (payload?.weight !== undefined && !isNumber(payload.weight)) {
    errors.push("weight must be a number");
  } else if (
    isNumber(payload?.weight) &&
    (payload.weight < 20 || payload.weight > 300)
  ) {
    errors.push("weight must be between 20 and 300 kg");
  }

  if (
    payload?.cycleRegularity !== undefined &&
    !["Regular", "Irregular"].includes(payload.cycleRegularity)
  ) {
    errors.push("cycleRegularity must be Regular or Irregular");
  }

  if (payload?.cycleLength !== undefined && !isNumber(payload.cycleLength)) {
    errors.push("cycleLength must be a number");
  } else if (
    isNumber(payload?.cycleLength) &&
    (payload.cycleLength < 15 || payload.cycleLength > 60)
  ) {
    errors.push("cycleLength must be between 15 and 60 days");
  }

  if (
    payload?.missedPeriods !== undefined &&
    !isNumber(payload.missedPeriods)
  ) {
    errors.push("missedPeriods must be a number");
  } else if (
    isNumber(payload?.missedPeriods) &&
    (payload.missedPeriods < 0 || payload.missedPeriods > 12)
  ) {
    errors.push("missedPeriods must be between 0 and 12");
  }

  if (payload?.heavyPeriods !== undefined && !isBoolean(payload.heavyPeriods)) {
    errors.push("heavyPeriods must be a boolean");
  }

  if (
    payload?.acneSeverity !== undefined &&
    !["None", "Mild", "Severe"].includes(payload.acneSeverity)
  ) {
    errors.push("acneSeverity must be None, Mild, or Severe");
  }

  [
    "excessHair",
    "hairLoss",
    "suddenWeightGain",
    "darkPatches",
    "familyHistory",
    "thyroidOrDiabetes",
  ].forEach((field) => {
    if (payload?.[field] !== undefined && !isBoolean(payload[field])) {
      errors.push(`${field} must be a boolean`);
    }
  });

  if (
    payload?.activityLevel !== undefined &&
    !["Low", "Moderate", "High"].includes(payload.activityLevel)
  ) {
    errors.push("activityLevel must be Low, Moderate, or High");
  }

  if (payload?.sleepHours !== undefined && !isNumber(payload.sleepHours)) {
    errors.push("sleepHours must be a number");
  } else if (
    isNumber(payload?.sleepHours) &&
    (payload.sleepHours < 0 || payload.sleepHours > 24)
  ) {
    errors.push("sleepHours must be between 0 and 24");
  }

  if (payload?.stressLevel !== undefined && !isNumber(payload.stressLevel)) {
    errors.push("stressLevel must be a number");
  } else if (
    isNumber(payload?.stressLevel) &&
    ![1, 2, 3, 4, 5].includes(payload.stressLevel)
  ) {
    errors.push("stressLevel must be between 1 and 5");
  }

  return errors;
};
