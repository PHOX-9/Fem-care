import { Request, Response } from "express";
import Onboarding from "./onboarding.model";
import User from "../../models/User";
import {
  calculateBmi,
  validateOnboardingPayload,
  OnboardingPayload,
} from "./onboarding.service";

export const createOnboarding = async (req: Request, res: Response) => {
  try {
    const userId = (req as any).userId;
    const payload = req.body as OnboardingPayload;

    const validationErrors = validateOnboardingPayload(payload);
    if (validationErrors.length > 0) {
      return res.status(400).json({
        message: "Validation failed",
        errors: validationErrors,
      });
    }

    const existing = await Onboarding.findOne({ user: userId });
    if (existing) {
      return res.status(409).json({
        message: "Onboarding already completed",
      });
    }

    const bmi = calculateBmi(payload.height, payload.weight);

    const onboarding = await Onboarding.create({
      user: userId,
      ...payload,
      bmi,
    });

    await User.findByIdAndUpdate(userId, {
      $set: { onboardingCompleted: true },
    });

    return res.status(201).json({
      message: "Onboarding saved",
      data: onboarding,
    });
  } catch (error) {
    if ((error as any)?.code === 11000) {
      return res.status(409).json({
        message: "Onboarding already completed",
      });
    }
    return res.status(500).json({ message: "Server error" });
  }
};

export const getOnboarding = async (req: Request, res: Response) => {
  try {
    const userId = (req as any).userId;

    const onboarding = await Onboarding.findOne({ user: userId }).lean();
    if (!onboarding) {
      return res.status(404).json({
        message: "Onboarding data not found",
      });
    }

    return res.json({ data: onboarding });
  } catch (error) {
    return res.status(500).json({ message: "Server error" });
  }
};
