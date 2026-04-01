# trained_model = trainer.train()

# # 最终测试
# print("\n" + "=" * 50)
# print("Final Testing")
# print("=" * 50)
# test_dataset = CodeDataset(test_hf, tokenizer)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # 使用训练集重新计算几何中位数用于测试
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# trainer.compute_geometric_median_prototypes(train_loader)

# _, test_metrics = trainer.evaluate(test_loader, epoch="final", save_prefix="test")

# print("\nFinal Test Results:")
# print(f"Binary MCC: {test_metrics['binary']['mcc']:.4f}")
# print(f"Global Macro F1: {test_metrics['global_macro']['f1']:.4f}")

# print(f"\nAll outputs saved to: {OUTPUT_DIR}")
