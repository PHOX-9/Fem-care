import mongoose, { Schema, Document, Types } from "mongoose";

export interface IOnboarding extends Document {
  user: Types.ObjectId;
  age: number;
  height: number;
  weight: number;
  bmi: number;
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
  createdAt: Date;
  updatedAt: Date;
}

const OnboardingSchema = new Schema<IOnboarding>(
  {
    user: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: true,
      unique: true,
      index: true,
    },
    age: { type: Number, required: true },
    height: { type: Number, required: true },
    weight: { type: Number, required: true },
    bmi: { type: Number, required: true },
    cycleRegularity: {
      type: String,
      required: true,
      enum: ["Regular", "Irregular"],
    },
    cycleLength: { type: Number, required: true },
    missedPeriods: { type: Number, required: true },
    heavyPeriods: { type: Boolean, required: true },
    acneSeverity: {
      type: String,
      required: true,
      enum: ["None", "Mild", "Severe"],
    },
    excessHair: { type: Boolean, required: true },
    hairLoss: { type: Boolean, required: true },
    suddenWeightGain: { type: Boolean, required: true },
    darkPatches: { type: Boolean, required: true },
    familyHistory: { type: Boolean, required: true },
    thyroidOrDiabetes: { type: Boolean, required: true },
    activityLevel: {
      type: String,
      required: true,
      enum: ["Low", "Moderate", "High"],
    },
    sleepHours: { type: Number, required: true },
    stressLevel: {
      type: Number,
      required: true,
      enum: [1, 2, 3, 4, 5],
    },
  },
  { timestamps: true },
);

export default mongoose.model<IOnboarding>("Onboarding", OnboardingSchema);
