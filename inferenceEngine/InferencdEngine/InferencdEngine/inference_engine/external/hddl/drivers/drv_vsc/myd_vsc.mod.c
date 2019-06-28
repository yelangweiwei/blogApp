#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

MODULE_INFO(vermagic, VERMAGIC_STRING);

__visible struct module __this_module
__attribute__((section(".gnu.linkonce.this_module"))) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

static const struct modversion_info ____versions[]
__used
__attribute__((section("__versions"))) = {
	{ 0x9d35aeec, __VMLINUX_SYMBOL_STR(module_layout) },
	{ 0x90766838, __VMLINUX_SYMBOL_STR(noop_llseek) },
	{ 0xcca0be3c, __VMLINUX_SYMBOL_STR(usb_deregister) },
	{ 0x2b626cd6, __VMLINUX_SYMBOL_STR(usb_register_driver) },
	{ 0xdb7305a1, __VMLINUX_SYMBOL_STR(__stack_chk_fail) },
	{ 0x4f8b5ddb, __VMLINUX_SYMBOL_STR(_copy_to_user) },
	{ 0xd62c833f, __VMLINUX_SYMBOL_STR(schedule_timeout) },
	{ 0xf08242c2, __VMLINUX_SYMBOL_STR(finish_wait) },
	{ 0x2207a57f, __VMLINUX_SYMBOL_STR(prepare_to_wait_event) },
	{ 0x1000e51, __VMLINUX_SYMBOL_STR(schedule) },
	{ 0xa1c76e0a, __VMLINUX_SYMBOL_STR(_cond_resched) },
	{ 0x7f02188f, __VMLINUX_SYMBOL_STR(__msecs_to_jiffies) },
	{ 0x477625f9, __VMLINUX_SYMBOL_STR(mutex_lock_interruptible) },
	{ 0xe7985cd8, __VMLINUX_SYMBOL_STR(usb_unanchor_urb) },
	{ 0x156a8a59, __VMLINUX_SYMBOL_STR(down_trylock) },
	{ 0x2810cbaf, __VMLINUX_SYMBOL_STR(usb_submit_urb) },
	{ 0xd5ac7f96, __VMLINUX_SYMBOL_STR(usb_anchor_urb) },
	{ 0x4f6b400b, __VMLINUX_SYMBOL_STR(_copy_from_user) },
	{ 0xb9d025c9, __VMLINUX_SYMBOL_STR(llist_del_first) },
	{ 0x6dc0c9dc, __VMLINUX_SYMBOL_STR(down_interruptible) },
	{ 0x78764f4e, __VMLINUX_SYMBOL_STR(pv_irq_ops) },
	{ 0xe5815f8a, __VMLINUX_SYMBOL_STR(_raw_spin_lock_irq) },
	{ 0x702d043, __VMLINUX_SYMBOL_STR(usb_deregister_dev) },
	{ 0x44d06b4, __VMLINUX_SYMBOL_STR(usb_autopm_put_interface) },
	{ 0xd9607266, __VMLINUX_SYMBOL_STR(_dev_info) },
	{ 0xd4ea0736, __VMLINUX_SYMBOL_STR(usb_register_dev) },
	{ 0x2da1b27c, __VMLINUX_SYMBOL_STR(usb_alloc_urb) },
	{ 0xd2b09ce5, __VMLINUX_SYMBOL_STR(__kmalloc) },
	{ 0x26b1b15b, __VMLINUX_SYMBOL_STR(usb_alloc_coherent) },
	{ 0xc375dec3, __VMLINUX_SYMBOL_STR(usb_get_dev) },
	{ 0x9e88526, __VMLINUX_SYMBOL_STR(__init_waitqueue_head) },
	{ 0x17d070c2, __VMLINUX_SYMBOL_STR(__mutex_init) },
	{ 0x2142697b, __VMLINUX_SYMBOL_STR(kmem_cache_alloc_trace) },
	{ 0x8a9809d6, __VMLINUX_SYMBOL_STR(kmalloc_caches) },
	{ 0xa6bbd805, __VMLINUX_SYMBOL_STR(__wake_up) },
	{ 0x78e739aa, __VMLINUX_SYMBOL_STR(up) },
	{ 0xc7a1840e, __VMLINUX_SYMBOL_STR(llist_add_batch) },
	{ 0x6bf1c17f, __VMLINUX_SYMBOL_STR(pv_lock_ops) },
	{ 0xe259ae9e, __VMLINUX_SYMBOL_STR(_raw_spin_lock) },
	{ 0x194651a6, __VMLINUX_SYMBOL_STR(dev_err) },
	{ 0x27e1a049, __VMLINUX_SYMBOL_STR(printk) },
	{ 0x16305289, __VMLINUX_SYMBOL_STR(warn_slowpath_null) },
	{ 0x26c2e913, __VMLINUX_SYMBOL_STR(usb_autopm_get_interface) },
	{ 0x6a1654ce, __VMLINUX_SYMBOL_STR(usb_find_interface) },
	{ 0x37a0cba, __VMLINUX_SYMBOL_STR(kfree) },
	{ 0xc0f333fa, __VMLINUX_SYMBOL_STR(usb_put_dev) },
	{ 0x809dd6c4, __VMLINUX_SYMBOL_STR(usb_free_urb) },
	{ 0xcf05e462, __VMLINUX_SYMBOL_STR(usb_init_urb) },
	{ 0x40584674, __VMLINUX_SYMBOL_STR(usb_free_coherent) },
	{ 0x11761f56, __VMLINUX_SYMBOL_STR(mutex_lock) },
	{ 0xb931ebc5, __VMLINUX_SYMBOL_STR(usb_kill_urb) },
	{ 0x4395953e, __VMLINUX_SYMBOL_STR(usb_kill_anchored_urbs) },
	{ 0x24f45195, __VMLINUX_SYMBOL_STR(usb_wait_anchor_empty_timeout) },
	{ 0x347cd1b3, __VMLINUX_SYMBOL_STR(mutex_unlock) },
	{ 0xbdfb6dbb, __VMLINUX_SYMBOL_STR(__fentry__) },
};

static const char __module_depends[]
__used
__attribute__((section(".modinfo"))) =
"depends=";

MODULE_ALIAS("usb:v03E7pF63Bd*dc*dsc*dp*ic*isc*ip*in*");

MODULE_INFO(srcversion, "57D9F2B07DD05FAB560B8A7");
