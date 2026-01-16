import { Router } from "express";
import authMiddleware from "../../middlewares/auth";
import { createOnboarding, getOnboarding } from "./onboarding.controller";

const router = Router();

router.post("/", authMiddleware, createOnboarding);
router.get("/", authMiddleware, getOnboarding);

export default router;
